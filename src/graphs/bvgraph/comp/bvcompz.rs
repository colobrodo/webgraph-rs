/*
 * SPDX-FileCopyrightText: 2023 Inria
 * SPDX-FileCopyrightText: 2023 Sebastiano Vigna
 *
 * SPDX-License-Identifier: Apache-2.0 OR LGPL-2.1-or-later
 */

use super::bvcomp::Compressor;
use crate::prelude::*;
use common_traits::Sequence;
use lender::prelude::*;

/// An Entry for the table used to save the intermediate computation
/// of the dynamic algorithm to select the best references.
/// It represents if a reference to a node, with a know amount of previous
/// references chain length, is choosen and how much less it costs to all his
/// referent with respect to compress the node without any selected reference.
#[derive(Default, Clone)]
struct ReferenceTableEntry {
    saved_cost: f32,
    choosen: bool,
}

/// A BvGraph compressor based on the approximation algorithm described in Zuckerli,
/// which is used to compress a graph into a BvGraph.   
/// This compressor uses a dynamic algorithm to find the best references and his
/// memory usage and precision can be traded off using the `chunk_size`.
#[derive(Debug, Clone)]
pub struct BvCompZ<E> {
    /// The ring-buffer that stores the neighbours of the last
    /// `compression_window` neighbours
    backrefs: CircularBuffer<Vec<usize>>,
    /// The references to the adjecency list to copy
    references: Vec<usize>,
    /// Stimated costs in saved bits using the current reference selection versus the extensive list   
    saved_costs: Vec<f32>,
    /// The number of nodes for which the reference selection algorithm is executed.
    /// Used in the dynamic algorithm to manage the tradeoff between memory consumption
    /// and space gained in compression.
    chunk_size: usize,
    /// The bitstream writer, this implements the mock function so we can
    /// do multiple tentative compressions and use the real one once we figured
    /// out how to compress the graph best
    encoder: E,
    /// When compressing we need to store metadata. So we store the compressors
    /// to reuse the allocations for perf reasons.
    compressors: Vec<Compressor>,
    /// The number of previous nodes that will be considered during the compression
    compression_window: usize,
    /// The maximum recursion depth that will be used to decompress a node
    max_ref_count: usize,
    /// The minimum length of sequences that will be compressed as a (start, len)
    min_interval_length: usize,
    /// The current node we are compressing
    curr_node: usize,
    /// The first node of the chunck in which the nodes' references are calculated together
    start_chunk_node: usize,
    /// The number of arcs compressed so far
    pub arcs: u64,
}

impl<E: EncodeAndEstimate> BvCompZ<E> {
    /// This value for `min_interval_length` implies that no intervalization will be performed.
    pub const NO_INTERVALS: usize = Compressor::NO_INTERVALS;

    /// Creates a new BvGraph compressor.
    pub fn new(
        encoder: E,
        compression_window: usize,
        chunk_size: usize,
        max_ref_count: usize,
        min_interval_length: usize,
        start_node: usize,
    ) -> Self {
        BvCompZ {
            backrefs: CircularBuffer::new(chunk_size + 1),
            references: Vec::with_capacity(chunk_size + 1),
            saved_costs: Vec::with_capacity(chunk_size + 1),
            chunk_size,
            encoder,
            min_interval_length,
            compression_window,
            max_ref_count,
            start_chunk_node: start_node,
            curr_node: start_node,
            compressors: (0..chunk_size + 1).map(|_| Compressor::new()).collect(),
            arcs: 0,
        }
    }

    /// Push a new node to the compressor.
    /// The iterator must yield the successors of the node and the nodes HAVE
    /// TO BE CONTIGUOUS (i.e. if a node has no neighbours you have to pass an
    /// empty iterator)
    pub fn push<I: IntoIterator<Item = usize>>(&mut self, succ_iter: I) -> anyhow::Result<u64> {
        // collect the iterator inside the backrefs, to reuse the capacity already
        // allocated
        {
            let mut succ_vec = self.backrefs.take(self.curr_node);
            succ_vec.clear();
            succ_vec.extend(succ_iter);
            self.backrefs.replace(self.curr_node, succ_vec);
        }
        // get the ref
        let curr_list = &self.backrefs[self.curr_node];
        self.arcs += curr_list.len() as u64;
        // first try to compress the current node without references
        let compressor = &mut self.compressors[0];
        // Compute how we would compress this
        compressor.compress(curr_list, None, self.min_interval_length)?;
        // avoid the mock writing
        if self.compression_window == 0 {
            let written_bits = compressor.write(
                &mut self.encoder,
                self.curr_node,
                None,
                self.min_interval_length,
            )?;
            // update the current node
            self.curr_node += 1;
            return Ok(written_bits);
        }
        // The delta of the best reference, by default 0 which is no compression
        let mut ref_delta = 0;
        let cost = {
            let mut estimator = self.encoder.estimator();
            // Write the compressed data
            compressor.write(
                &mut estimator,
                self.curr_node,
                Some(0),
                self.min_interval_length,
            )?
        };
        let mut saved_cost = 0;
        let mut min_bits = cost;

        let deltas = 1 + self
            .compression_window
            .min(self.curr_node - self.start_chunk_node);
        // compression windows is not zero, so compress the current node
        for delta in 1..deltas {
            let ref_node = self.curr_node - delta;
            // Get the neighbours of this previous len_zetanode
            let ref_list = &self.backrefs[ref_node];
            // No neighbours, no compression
            if ref_list.is_empty() {
                continue;
            }
            // Get its compressor
            let compressor = &mut self.compressors[delta];
            // Compute how we would compress this
            compressor.compress(curr_list, Some(ref_list), self.min_interval_length)?;
            // Compute how many bits it would use, using the mock writer
            let bits = {
                let mut estimator = self.encoder.estimator();
                compressor.write(
                    &mut estimator,
                    self.curr_node,
                    Some(delta),
                    self.min_interval_length,
                )?
            };
            // keep track of the best, it's strictly less so we keep the
            // nearest one in the case of multiple equal ones
            if bits < min_bits {
                saved_cost = cost - bits;
                min_bits = bits;
                ref_delta = delta;
            }
        }
        // consistency check
        assert_eq!(
            self.references.len(),
            self.curr_node - self.start_chunk_node
        );
        self.saved_costs.push(saved_cost as f32);
        self.references.push(ref_delta);
        // update the current node
        let mut written_bits = 0;
        self.curr_node += 1;
        if self.references.len() >= self.chunk_size {
            written_bits = self.calculate_reference_selection()?;
        }
        Ok(written_bits)
    }

    /// Given an iterator over the nodes successors iterators, push them all.
    /// The iterator must yield the successors of the node and the nodes HAVE
    /// TO BE CONTIGUOUS (i.e. if a node has no neighbours you have to pass an
    /// empty iterator).
    ///
    /// This most commonly is called with a reference to a graph.
    pub fn extend<L>(&mut self, iter_nodes: L) -> anyhow::Result<u64>
    where
        L: IntoLender,
        L::Lender: for<'next> NodeLabelsLender<'next, Label = usize>,
    {
        let mut count = 0;
        for_! ( (_, succ) in iter_nodes {
            count += self.push(succ.into_iter())?;
        });
        // WAS
        // iter_nodes.for_each(|(_, succ)| self.push(succ)).sum()
        Ok(count)
    }

    fn calculate_reference_selection(&mut self) -> anyhow::Result<u64> {
        let n = self.references.len();
        self.update_references_for_max_lenght();
        assert_eq!(n, self.curr_node - self.start_chunk_node);
        assert_eq!(self.start_chunk_node, self.curr_node - n);

        // TODO: complete zuckerli algorithm using greedy selection of the nodes
        // iterate over all the backrefs and write them
        // calculate length of previous references' chains
        let mut chain_length = vec![0usize; self.chunk_size];
        for i in 0..n {
            if self.references[i] != 0 {
                let parent = i - self.references[i];
                chain_length[i] = chain_length[parent] + 1;
            }
        }
        // calculate the length of nexts reference chain
        let mut forward_chain_length = vec![0usize; self.chunk_size];
        for i in (0..n).rev() {
            if self.references[i] != 0 {
                // check if the subsequent length of my chain is greater than the one of
                // other children of my parent
                let parent = i - self.references[i];
                forward_chain_length[parent] =
                    forward_chain_length[parent].max(forward_chain_length[i] + 1);
            }
        }
        for i in 0..n {
            let node_index = self.curr_node - n + i;
            // recalculate the chain lenght because the reference can be changed
            // after a greedy re-add in a previouse iteration
            if self.references[i] != 0 {
                let parent = i - self.references[i];
                chain_length[i] = chain_length[parent] + 1;
            }
            let curr_list = &self.backrefs[node_index];
            // first try to compress the current node without references
            let compressor = &mut self.compressors[i];
            // Compute how we would compress this
            compressor.compress(curr_list, None, self.min_interval_length)?;
            let mut min_bits = {
                let mut estimator = self.encoder.estimator();
                // Write the compressed data
                compressor.write(
                    &mut estimator,
                    node_index,
                    Some(0),
                    self.min_interval_length,
                )?
            };

            let deltas = 1 + self.compression_window.min(i);
            // compression windows is not zero, so compress the current node
            for delta in 1..deltas {
                if chain_length[i - delta] + forward_chain_length[i] + 1 > self.max_ref_count {
                    continue;
                }
                let reference_index = node_index - delta;
                // Get the neighbours of this previous len_zetanode
                let ref_list = &self.backrefs[reference_index];
                // No neighbours, no compression
                if ref_list.is_empty() {
                    continue;
                }
                // Get its compressor
                let compressor = &mut self.compressors[i - delta];
                // Compute how we would compress this
                compressor.compress(curr_list, Some(ref_list), self.min_interval_length)?;
                // Compute how many bits it would use, using the mock writer
                let bits = {
                    let mut estimator = self.encoder.estimator();
                    compressor.write(
                        &mut estimator,
                        node_index,
                        Some(delta),
                        self.min_interval_length,
                    )?
                };
                // keep track of the best, it's strictly less so we keep the
                // nearest one in the case of multiple equal ones
                if bits < min_bits {
                    min_bits = bits;
                    self.references[i] = delta;
                }
            }
            if self.references[i] != 0 {
                let parent = i - self.references[i];
                chain_length[i] = chain_length[parent] + 1;
            }
        }

        let mut written_bits = 0;
        for i in 0..n {
            let node_index = self.curr_node - n + i;
            let curr_list = &self.backrefs[node_index];
            let reference = self.references[i];
            let ref_list = if reference == 0 {
                None
            } else {
                let reference_index = node_index - reference;
                Some(self.backrefs[reference_index].as_slice()).filter(|list| !list.is_empty())
            };
            let compressor = &mut self.compressors[i];
            compressor.compress(curr_list, ref_list, self.min_interval_length)?;
            written_bits += compressor.write(
                &mut self.encoder,
                node_index,
                Some(reference),
                self.min_interval_length,
            )?;
        }
        // reset the chunk starting point
        self.start_chunk_node = self.curr_node;
        // clear the refs array and the backrefs
        self.references.clear();
        self.saved_costs.clear();
        Ok(written_bits)
    }

    pub fn update_references_for_max_lenght(&mut self) {
        // consistency checks
        let n = self.references.len();
        debug_assert!(self.saved_costs.len() == n);
        for i in 0..n {
            debug_assert!(self.references[i] <= i);
            debug_assert!(self.saved_costs[i] >= 0.0);
            if self.references[i] == 0 {
                debug_assert!(self.saved_costs[i] == 0.0);
            }
        }

        // dag of nodes that points to the i-th element of the vector
        let mut out_edges: Vec<Vec<usize>> = vec![Vec::new(); n];
        for (i, reference) in self.references.iter().enumerate() {
            // 0 <= references[i] <= windows_size
            if reference != 0 {
                // for each j in out_edges, for each i in out_edges[j]: j + window_size >= i
                // circular buffer for the table but later we iterate in reverse order so we should always
                // keep at least the references list
                out_edges[i - reference].push(i);
            }
        }
        // table for dynamic programming: the maximum weight of the subforest rooted in x
        // that has no paths longer than max_lenght and where the root x
        // is not part of a path longer than i (from 0..max_lenght)
        // so using dyn[node][max_length] denotes the weight where
        // we are considering node to be the root
        let mut dyn_table = vec![vec![ReferenceTableEntry::default(); self.max_ref_count + 1]; n];

        for i in (0..n).rev() {
            // in the paper M_r(i) so the case where I don't choose this node to be referred from other lists
            // and favor the children so they can be the end of full (max_lenght) reference chains
            let mut child_sum_full_chain = 0.0;
            for child in out_edges[i].iter() {
                child_sum_full_chain += dyn_table[child][self.max_ref_count].saved_cost;
            }

            dyn_table[i][0] = ReferenceTableEntry {
                saved_cost: child_sum_full_chain,
                choosen: false,
            };

            // counting parent link, if any.
            for links_to_use in 1..=self.max_ref_count {
                // Now we are choosing i to have at most children chains of 'links_to_use'
                // (because we used 'max_length - links_to_use' links before somewhere)
                let mut child_sum = self.saved_costs[i];
                // Take it.
                for child in out_edges[i].iter() {
                    child_sum += dyn_table[child][links_to_use - 1].saved_cost;
                }
                dyn_table[i][links_to_use] = if child_sum > child_sum_full_chain {
                    ReferenceTableEntry {
                        saved_cost: child_sum,
                        choosen: true,
                    }
                } else {
                    ReferenceTableEntry {
                        saved_cost: child_sum_full_chain,
                        choosen: false,
                    }
                };
            }
        }

        let mut available_length = vec![self.max_ref_count; n];
        // always choose the maximum available lengths calculated in the previous step
        for i in 0..self.references.len() {
            if dyn_table[i][available_length[i]].choosen {
                // Taken: push available_length.
                for child in out_edges[i].iter() {
                    available_length[child] = available_length[i] - 1;
                }
            } else {
                // Not taken: remove reference.
                self.references[i] = 0;
            }
        }
    }

    /// Consume the compressor return the number of bits written by
    /// flushing the encoder (0 for instantaneous codes)
    pub fn flush(mut self) -> Result<usize, E::Error> {
        // TODO: convert anyhow error
        let remaining_chunck_bits = if self.compression_window > 0 {
            self.calculate_reference_selection().unwrap()
        } else {
            0
        };
        let flushed = self.encoder.flush()?;
        Ok(remaining_chunck_bits as usize + flushed)
    }
}

#[cfg(test)]
mod test {

    use self::sequential::Iter;

    use super::*;
    use dsi_bitstream::prelude::*;
    use itertools::Itertools;
    use std::fs::File;
    use std::io::{BufReader, BufWriter};

    #[test]
    fn test_writer_window_zero() -> anyhow::Result<()> {
        test_compression(0, 0)?;
        test_compression(0, 1)?;
        test_compression(0, 2)?;
        Ok(())
    }

    #[test]
    fn test_writer_window_one() -> anyhow::Result<()> {
        test_compression(1, 0)?;
        test_compression(1, 1)?;
        test_compression(1, 2)?;
        Ok(())
    }

    #[test]
    fn test_writer_window_two() -> anyhow::Result<()> {
        test_compression(2, 0)?;
        test_compression(2, 1)?;
        test_compression(2, 2)?;
        Ok(())
    }

    #[test]
    fn test_writer_cnr() -> anyhow::Result<()> {
        let compression_window = 32;
        let min_interval_length = 4;

        let seq_graph = BvGraphSeq::with_basename("tests/data/cnr-2000")
            .endianness::<BE>()
            .load()?;

        // Compress the graph
        let file_path = "tests/data/cnr-2000.bvcompz";
        let bit_write = <BufBitWriter<BE, _>>::new(<WordAdapter<usize, _>>::new(BufWriter::new(
            File::create(file_path)?,
        )));

        let comp_flags = CompFlags {
            ..Default::default()
        };

        let codes_writer = <ConstCodesEncoder<BE, _>>::new(bit_write);

        let mut bvcomp = BvCompZ::new(
            codes_writer,
            compression_window,
            1000,
            3,
            min_interval_length,
            0,
        );

        bvcomp.extend(&seq_graph).unwrap();
        bvcomp.flush()?;

        // Read it back

        let bit_read = <BufBitReader<BE, _>>::new(<WordAdapter<u32, _>>::new(BufReader::new(
            File::open(file_path)?,
        )));

        //let codes_reader = <DynamicCodesReader<LE, _>>::new(bit_read, &comp_flags)?;
        let codes_reader = <ConstCodesDecoder<BE, _>>::new(bit_read, &comp_flags)?;

        let mut seq_iter = Iter::new(
            codes_reader,
            seq_graph.num_nodes(),
            compression_window,
            min_interval_length,
        );
        // Check that the graph is the same
        let mut iter = seq_graph.iter().enumerate();
        while let Some((i, (true_node_id, true_succ))) = iter.next() {
            let (seq_node_id, seq_succ) = seq_iter.next().unwrap();

            assert_eq!(true_node_id, i);
            assert_eq!(true_node_id, seq_node_id);
            assert_eq!(
                true_succ.collect_vec(),
                seq_succ.into_iter().collect_vec(),
                "node_id: {}",
                i
            );
        }
        std::fs::remove_file(file_path).unwrap();

        Ok(())
    }

    fn test_compression(
        compression_window: usize,
        min_interval_length: usize,
    ) -> anyhow::Result<()> {
        let seq_graph = BvGraphSeq::with_basename("tests/data/cnr-2000")
            .endianness::<BE>()
            .load()?;

        // Compress the graph
        let mut buffer: Vec<u64> = Vec::new();
        let bit_write = <BufBitWriter<LE, _>>::new(MemWordWriterVec::new(&mut buffer));

        let comp_flags = CompFlags {
            ..Default::default()
        };

        let codes_writer = <ConstCodesEncoder<LE, _>>::new(bit_write);

        let max_ref_count = 3;
        let mut bvcomp = BvCompZ::new(
            codes_writer,
            compression_window,
            10000,
            max_ref_count,
            min_interval_length,
            0,
        );

        bvcomp.extend(&seq_graph).unwrap();
        bvcomp.flush()?;

        // Read it back
        let buffer_32: &[u32] = unsafe { buffer.align_to().1 };
        let bit_read = <BufBitReader<LE, _>>::new(MemWordReader::new(buffer_32));

        //let codes_reader = <DynamicCodesReader<LE, _>>::new(bit_read, &comp_flags)?;
        let codes_reader = <ConstCodesDecoder<LE, _>>::new(bit_read, &comp_flags)?;

        let mut seq_iter = Iter::new(
            codes_reader,
            seq_graph.num_nodes(),
            compression_window,
            min_interval_length,
        );
        // Check that the graph is the same
        let mut iter = seq_graph.iter().enumerate();
        while let Some((i, (true_node_id, true_succ))) = iter.next() {
            let (seq_node_id, seq_succ) = seq_iter.next().unwrap();

            assert_eq!(true_node_id, i);
            assert_eq!(true_node_id, seq_node_id);
            assert_eq!(
                true_succ.collect_vec(),
                seq_succ.collect_vec(),
                "node_id: {}",
                i
            );
        }

        Ok(())
    }
}
