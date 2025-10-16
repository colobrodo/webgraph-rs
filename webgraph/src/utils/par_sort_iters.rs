/*
 * SPDX-FileCopyrightText: 2025 Inria
 * SPDX-FileCopyrightText: 2025 Sebastiano Vigna
 *
 * SPDX-License-Identifier: Apache-2.0 OR LGPL-2.1-or-later
 */

#![allow(clippy::type_complexity)]

//! Facilities to sort in parallel externally (labelled) pairs of nodes
//! returned by a sequence of iterators.
//!
//! The typical use of [`ParSortIters`] is to sort (labelled) pairs of nodes
//! representing a (labelled) graph; the resulting [`SplitIters`] structure can
//! be then used to build a compressed representation of the graph using, for
//! example,
//! [`BvComp::parallel_iter`](crate::graphs::bvgraph::BvComp::parallel_iter).
//!
//! For example, when transposing or permuting a
//! [splittable](crate::traits::SplitLabeling) graph one obtains such a sequence
//! of iterators.
//!
//! If your pairs are emitted by a single parallel iterator, consider using
//! [`ParSortPairs`](crate::utils::par_sort_pairs::ParSortPairs) instead.

use std::marker::PhantomData;
use std::num::NonZeroUsize;
use sync_cell_slice::SyncSlice;

use anyhow::{Context, Result};
use dsi_bitstream::traits::NE;
use dsi_progress_logger::{concurrent_progress_logger, ProgressLog};
use rayon::prelude::*;

use super::sort_pairs::{BatchIterator, BitReader, BitWriter, KMergeIters, Triple};
use super::MemoryUsage;
use crate::traits::{BitDeserializer, BitSerializer};
use crate::utils::SplitIters;

/// Takes a sequence of iterators of (labelled)pairs as input, and turns them
/// into [`SplitIters`] structure which is suitable for
/// [`BvComp::parallel_iter`](crate::graphs::bvgraph::BvComp::parallel_iter).
///
/// Note that batches will be memory-mapped. If you encounter OS-level errors
/// using this class (e.g., `ENOMEM: Out of memory` under Linux), please review
/// the limitations of your OS regarding memory-mapping (e.g.,
/// `/proc/sys/vm/max_map_count` under Linux).
///
/// # Examples
///
/// In this example we transpose a graph in parallel by splitting it, exchanging
/// the source and destination of each arc, sorting the resulting pairs in
/// parallel using [`ParSortIters`], and then compressing the result using
/// [`BvComp::parallel_iter`](crate::graphs::bvgraph::BvComp::parallel_iter):
///
/// ```
/// use std::num::NonZeroUsize;
///
/// use dsi_bitstream::traits::BigEndian;
/// use rayon::prelude::*;
/// use webgraph::prelude::*;
/// use webgraph::graphs::bvgraph::{BvComp, CompFlags};
/// use webgraph::traits::{SequentialLabeling, SplitLabeling};
/// use webgraph::utils::par_sort_iters::ParSortIters;
///
/// // Build a small VecGraph
/// let g = VecGraph::from_arcs([
///     (0, 4),
///     (1, 0),
///     (1, 3),
///     (2, 1),
///     (3, 2),
/// ]);
///
/// let num_nodes = g.num_nodes();
/// let num_partitions = 2;
///
/// // Split the graph into lenders and convert each to pairs
/// let pairs: Vec<_> = g
///     .split_iter(num_partitions)
///     .into_iter()
///     .map(|(_start_node, lender)| lender.into_pairs().map(|(src, dst)| (dst, src)))
///     .collect();
///
/// // Sort the pairs using ParSortIters
/// let pair_sorter = ParSortIters::new(num_nodes)?
///     .num_partitions(NonZeroUsize::new(num_partitions).unwrap());
///
/// let sorted = pair_sorter.sort(pairs)?;
///
/// // Convert to (node, lender) pairs using From trait
/// let pairs: Vec<_> = sorted.into();
///
/// // Compress in parallel using BvComp::parallel_iter
/// let bvcomp_tmp_dir = tempfile::tempdir()?;
/// let bvcomp_out_dir = tempfile::tempdir()?;
///
/// BvComp::parallel_iter::<BigEndian, _>(
///     &bvcomp_out_dir.path().join("graph"),
///     pairs.into_iter(),
///     num_nodes,
///     CompFlags::default(),
///     &rayon::ThreadPoolBuilder::default().build()?,
///     bvcomp_tmp_dir.path(),
/// )?;
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
pub struct ParSortIters<L = ()> {
    num_nodes: usize,
    expected_num_pairs: Option<usize>,
    num_partitions: NonZeroUsize,
    memory_usage: MemoryUsage,
    marker: PhantomData<L>,
}

impl ParSortIters<()> {
    /// See [`try_sort`](ParSortIters::try_sort).
    ///
    /// This is a convenience method for iterators that cannot fail.
    pub fn sort(
        &self,
        pairs: impl IntoIterator<
            Item: IntoIterator<Item = (usize, usize), IntoIter: Send> + Send,
            IntoIter: ExactSizeIterator,
        >,
    ) -> Result<SplitIters<impl IntoIterator<Item = (usize, usize), IntoIter: Send + Sync>>> {
        self.try_sort::<std::convert::Infallible>(pairs)
    }

    /// Sorts the output of the provided sequence of iterators, returning a
    /// [`SplitIters`] structure.
    pub fn try_sort<E: Into<anyhow::Error>>(
        &self,
        pairs: impl IntoIterator<
            Item: IntoIterator<Item = (usize, usize), IntoIter: Send> + Send,
            IntoIter: ExactSizeIterator,
        >,
    ) -> Result<SplitIters<impl IntoIterator<Item = (usize, usize), IntoIter: Send + Sync>>> {
        let split = <ParSortIters<()>>::try_sort_labeled::<(), (), E>(
            self,
            &(),
            (),
            pairs
                .into_iter()
                .map(|iter| iter.into_iter().map(|(src, dst)| (src, dst, ()))),
        )?;

        let iters_without_labels: Vec<_> = split
            .iters
            .into_vec()
            .into_iter()
            .map(|iter| iter.into_iter().map(|(src, dst, ())| (src, dst)))
            .collect();

        Ok(SplitIters::new(
            split.boundaries,
            iters_without_labels.into_boxed_slice(),
        ))
    }
}

impl<L> ParSortIters<L> {
    /// Creates a new [`ParSortIters`] instance.
    ///
    /// The methods [`num_partitions`](ParSortIters::num_partitions) (which sets
    /// the number of iterators in the resulting [`SplitIters`]),
    /// [`memory_usage`](ParSortIters::memory_usage), and
    /// [`expected_num_pairs`](ParSortIters::expected_num_pairs) can be used to
    /// customize the instance.
    ///
    /// This method will return an error if the number of CPUs
    /// returned by [`num_cpus::get()`](num_cpus::get()) is zero.
    pub fn new(num_nodes: usize) -> Result<Self> {
        Ok(Self {
            num_nodes,
            expected_num_pairs: None,
            num_partitions: NonZeroUsize::new(num_cpus::get()).context("zero CPUs")?,
            memory_usage: MemoryUsage::default(),
            marker: PhantomData,
        })
    }

    /// Approximate number of pairs to be sorted.
    ///
    /// Used only for progress reporting.
    pub fn expected_num_pairs(self, expected_num_pairs: usize) -> Self {
        Self {
            expected_num_pairs: Some(expected_num_pairs),
            ..self
        }
    }

    /// How many partitions to split the nodes into.
    ///
    /// This is the number of iterators in the resulting [`SplitIters`].
    ///
    /// Defaults to `num_cpus::get()`.
    pub fn num_partitions(self, num_partitions: NonZeroUsize) -> Self {
        Self {
            num_partitions,
            ..self
        }
    }

    /// How much memory to use for in-memory sorts.
    ///
    /// Larger values yield faster merges (by reducing logarithmically the
    /// number of batches to merge) but consume linearly more memory. We suggest
    /// to set this parameter as large as possible, depending on the available
    /// memory. The default is the default of [`MemoryUsage`].
    pub fn memory_usage(self, memory_usage: MemoryUsage) -> Self {
        Self {
            memory_usage,
            ..self
        }
    }

    /// See [`try_sort_labeled`](ParSortIters::try_sort_labeled).
    ///
    /// This is a convenience method for iterators that cannot fail.
    pub fn sort_labeled<S, D>(
        &self,
        serializer: &S,
        deserializer: D,
        pairs: impl IntoIterator<
            Item: IntoIterator<Item = (usize, usize, L), IntoIter: Send> + Send,
            IntoIter: ExactSizeIterator,
        >,
    ) -> Result<
        SplitIters<
            impl IntoIterator<
                Item = (
                    usize,
                    usize,
                    <D as BitDeserializer<NE, BitReader>>::DeserType,
                ),
                IntoIter: Send + Sync,
            >,
        >,
    >
    where
        L: Copy + Send + Sync,
        S: Sync + BitSerializer<NE, BitWriter, SerType = L>,
        D: Clone + Send + Sync + BitDeserializer<NE, BitReader, DeserType: Copy + Send + Sync>,
    {
        self.try_sort_labeled::<S, D, std::convert::Infallible>(serializer, deserializer, pairs)
    }

    /// Sorts the output of the provided sequence of iterators of (labelled)
    /// pairs, returning a [`SplitIters`] structure.
    ///
    /// This  method accept as type parameter a [`BitSerializer`] and a
    /// [`BitDeserializer`] that are used to serialize and deserialize the labels.
    ///
    /// The bit deserializer must be [`Clone`] because we need one for each
    /// [`BatchIterator`], and there are possible scenarios in which the
    /// deserializer might be stateful.
    pub fn try_sort_labeled<S, D, E: Into<anyhow::Error>>(
        &self,
        serializer: &S,
        deserializer: D,
        pairs: impl IntoIterator<
            Item: IntoIterator<Item = (usize, usize, L), IntoIter: Send> + Send,
            IntoIter: ExactSizeIterator,
        >,
    ) -> Result<
        SplitIters<
            impl IntoIterator<
                Item = (
                    usize,
                    usize,
                    <D as BitDeserializer<NE, BitReader>>::DeserType,
                ),
                IntoIter: Send + Sync,
            >,
        >,
    >
    where
        L: Copy + Send + Sync,
        S: Sync + BitSerializer<NE, BitWriter, SerType = L>,
        D: Clone + Send + Sync + BitDeserializer<NE, BitReader, DeserType: Copy + Send + Sync>,
    {
        let unsorted_pairs = pairs;

        let num_partitions = self.num_partitions.into();
        let num_buffers = rayon::current_num_threads() * num_partitions;
        let batch_size = self
            .memory_usage
            .batch_size::<Triple<L>>()
            .div_ceil(num_buffers);
        let num_nodes_per_partition = self.num_nodes.div_ceil(num_partitions);

        let mut pl = concurrent_progress_logger!(
            display_memory = true,
            item_name = "pair",
            local_speed = true,
            expected_updates = self.expected_num_pairs,
        );
        pl.start("Reading and sorting pairs");
        pl.info(format_args!("Per-processor batch size: {}", batch_size));

        let presort_tmp_dir =
            tempfile::tempdir().context("Could not create temporary directory")?;

        let unsorted_pairs = unsorted_pairs.into_iter();
        let num_blocks = unsorted_pairs.len();

        let mut partitioned_presorted_pairs = vec![vec![]; num_blocks];
        let result = partitioned_presorted_pairs.as_sync_slice();

        std::thread::scope(|s| {
            let presort_tmp_dir = &presort_tmp_dir;
            for (block_id, pair) in unsorted_pairs.enumerate() {
                let deserializer = deserializer.clone();
                let mut pl = pl.clone();
                s.spawn(move || {
                    let mut unsorted_buffers = (0..num_partitions)
                        .map(|_| Vec::with_capacity(batch_size))
                        .collect::<Vec<_>>();
                    let mut sorted_pairs =
                        (0..num_partitions).map(|_| Vec::new()).collect::<Vec<_>>();

                    for (src, dst, label) in pair {
                        let partition_id = src / num_nodes_per_partition;

                        let sorted_pairs = &mut sorted_pairs[partition_id];
                        let buf = &mut unsorted_buffers[partition_id];
                        if buf.len() >= buf.capacity() {
                            let buf_len = buf.len();
                            super::par_sort_pairs::flush_buffer(
                                presort_tmp_dir.path(),
                                serializer,
                                deserializer.clone(),
                                block_id,
                                partition_id,
                                sorted_pairs,
                                buf,
                            )
                            .context("Could not flush buffer")
                            .unwrap();
                            assert!(buf.is_empty(), "flush_buffer did not empty the buffer");
                            pl.update_with_count(buf_len);
                        }

                        buf.push(Triple {
                            pair: [src, dst],
                            label,
                        });
                    }

                    for (partition_id, (pairs, mut buf)) in sorted_pairs
                        .iter_mut()
                        .zip(unsorted_buffers.into_iter())
                        .enumerate()
                    {
                        let buf_len = buf.len();
                        super::par_sort_pairs::flush_buffer(
                            presort_tmp_dir.path(),
                            serializer,
                            deserializer.clone(),
                            block_id,
                            partition_id,
                            pairs,
                            &mut buf,
                        )
                        .context("Could not flush buffer at the end")
                        .unwrap();
                        assert!(buf.is_empty(), "flush_buffer did not empty the buffer");
                        pl.update_with_count(buf_len);
                    }

                    unsafe {
                        result[block_id].set(sorted_pairs);
                    }
                });
            }
        });

        // At this point, the iterator could be collected into
        // {worker_id -> {partition_id -> [iterators]}}
        // ie. Vec<Vec<Vec<BatchIterator>>>>.
        //
        // Let's merge the {partition_id -> [iterators]} maps of each worker
        let partitioned_presorted_pairs = partitioned_presorted_pairs.into_par_iter().reduce(
            || (0..num_partitions).map(|_| Vec::new()).collect(),
            |mut pair_partitions1: Vec<Vec<BatchIterator<D>>>,
             pair_partitions2: Vec<Vec<BatchIterator<D>>>|
             -> Vec<Vec<BatchIterator<D>>> {
                assert_eq!(pair_partitions1.len(), num_partitions);
                assert_eq!(pair_partitions2.len(), num_partitions);
                for (partition1, partition2) in pair_partitions1
                    .iter_mut()
                    .zip(pair_partitions2.into_iter())
                {
                    partition1.extend(partition2.into_iter());
                }
                pair_partitions1
            },
        );
        // At this point, the iterator was turned into
        // {partition_id -> [iterators]}
        // ie. Vec<Vec<BatchIterator>>>.
        pl.done();

        // Build boundaries array: [0, nodes_per_partition, 2*nodes_per_partition, ..., num_nodes]
        let boundaries: Vec<usize> = (0..=num_partitions)
            .map(|i| (i * num_nodes_per_partition).min(self.num_nodes))
            .collect();

        // Build iterators array
        let iters: Vec<_> = partitioned_presorted_pairs
            .into_iter()
            .map(|partition| {
                // 'partition' contains N iterators that are not sorted with respect to each other.
                // We merge them and turn them into a single sorted iterator.
                KMergeIters::new(partition)
            })
            .collect();

        Ok(SplitIters::new(
            boundaries.into_boxed_slice(),
            iters.into_boxed_slice(),
        ))
    }
}
