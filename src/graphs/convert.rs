use anyhow::{Context, Result};
use dsi_bitstream::prelude::*;
use dsi_progress_logger::{concurrent_progress_logger, prelude::*};
use lender::*;
use rayon::iter::{IntoParallelIterator, ParallelBridge, ParallelIterator};
use std::fs::File;
use std::io::{self, BufReader, BufWriter};
use std::path::{Path, PathBuf};
use std::time::Duration;

use webgraph::prelude::*;

use crate::huffman::*;

use super::estimator::{FixedEstimator, HuffmanEstimator, Log2Estimator};
use super::huffman_graph_encoder::{HuffmanGraphEncoder, HuffmanGraphEncoderBuilder};
use super::CompressorType;
use super::Estimator;
use super::{CompressionParameters, ContextModel};

/// Type alias for an encoder builder using `HuffmanEstimator` with an owned `CostModel`.
type HuffmanEstimatedEncoderBuilder<EP, C> =
    HuffmanGraphEncoderBuilder<HuffmanEstimator<EP, CostModel<EP>, C>, C, EP>;

/// Result of compressing a chunk of the graph in parallel.
/// Contains metadata about the compressed chunk files and their sizes.
#[derive(Debug)]
struct ChunkCompressionResult {
    thread_id: usize,
    first_node: usize,
    last_node: usize,
    chunk_dir: PathBuf,
    graph_written_bits: u64,
    offsets_written_bits: u64,
    num_arcs: u64,
}

/// A job representing a chunk of the graph to be compressed in parallel.
/// Contains all data needed by a worker thread to compress its assigned chunk.
struct ChunkedCompressionJob<L, EP: EncodeParams, E> {
    thread_id: usize,
    thread_lender: L,
    encoder: HuffmanEncoder<EP>,
    chunk_dir: PathBuf,
    compression_params: CompressionParameters,
    estimator: E,
}

/// A factory trait for creating thread-local estimators of a specific encoders.
///
/// This trait is used to instantiate estimators in parallel contexts where each thread
/// needs its own independent estimator instance of type `E`.
trait ThreadEstimatorFactory<'a, E: Encode + Send + Sync> {
    fn create_estimator(&self) -> E;
}

/// Factory that creates default-constructed estimators for parallel compression.
struct DefaultEstimatorFactory<E: Encode + Send + Sync + Default> {
    _marker: core::marker::PhantomData<E>,
}

impl<'a, E: Encode + Send + Sync + Default> ThreadEstimatorFactory<'a, E>
    for DefaultEstimatorFactory<E>
{
    fn create_estimator(&self) -> E {
        E::default()
    }
}

impl<E: Encode + Send + Sync + Default> Default for DefaultEstimatorFactory<E> {
    fn default() -> Self {
        Self {
            _marker: std::marker::PhantomData,
        }
    }
}

/// Factory for creating `HuffmanEstimator` instances with a captured cost model reference.
/// This factory captures a reference to the cost model from a previous estimation round and creates
/// thread-local `HuffmanEstimator` instances for parallel compression.
struct HuffmanEstimatorFactory<'a, EP: EncodeParams, C: ContextModel> {
    cost_model: &'a CostModel<EP>,
    _marker: core::marker::PhantomData<C>,
}

impl<'a, EP: EncodeParams + Send + Sync, C: ContextModel + Default + Copy + Send + Sync>
    ThreadEstimatorFactory<'a, HuffmanEstimator<EP, &'a CostModel<EP>, C>>
    for HuffmanEstimatorFactory<'a, EP, C>
{
    fn create_estimator(&self) -> HuffmanEstimator<EP, &'a CostModel<EP>, C> {
        HuffmanEstimator::new(self.cost_model, C::default())
    }
}

/// Run one reference-selection pass: collect symbols with the given estimator.
/// Returns a builder seeded with frequency estimates for the next stage.
fn reference_selection_round<
    G: SequentialGraph,
    EP: EncodeParams,
    E: Encode,
    C: ContextModel + Default,
>(
    graph: &G,
    huffman_graph_encoder_builder: HuffmanGraphEncoderBuilder<E, C, EP>,
    compression_parameters: &CompressionParameters,
    msg: impl AsRef<str>,
    pl: &mut ConcurrentWrapper,
) -> Result<HuffmanEstimatedEncoderBuilder<EP, C>> {
    let num_symbols = 1 << compression_parameters.max_bits;
    let huffman_estimator = huffman_graph_encoder_builder.build_estimator();
    // setup for the new iteration with huffman estimator
    let mut huffman_graph_encoder_builder =
        HuffmanGraphEncoderBuilder::<_, _, EP>::new(num_symbols, huffman_estimator, C::default());
    // discard all the offsets
    let offsets_writer = OffsetsWriter::from_write(io::empty(), false)?;
    pl.item_name("node")
        .expected_updates(Some(graph.num_nodes()));
    pl.start(msg);
    match compression_parameters.compressor {
        CompressorType::Approximated { chunk_size } => {
            let mut compressor = BvCompZ::new(
                &mut huffman_graph_encoder_builder,
                offsets_writer,
                compression_parameters.compression_window,
                chunk_size,
                compression_parameters.max_ref_count,
                compression_parameters.min_interval_length,
                0,
            );
            for_![ (_, succ) in graph.iter() {
                compressor.push(succ)?;
                pl.update();
            }];
            compressor.flush()?;
        }
        CompressorType::Greedy => {
            let mut compressor = BvComp::new(
                &mut huffman_graph_encoder_builder,
                offsets_writer,
                compression_parameters.compression_window,
                compression_parameters.max_ref_count,
                compression_parameters.min_interval_length,
                0,
            );
            for_![ (_, succ) in graph.iter() {
                compressor.push(succ)?;
                pl.update();
            }];
            compressor.flush()?;
        }
    }
    pl.done();
    Ok(huffman_graph_encoder_builder)
}

/// Generic helper for parallel compression: it iterates over a split lender and computes the symbols' frequencies
/// with the given estimator factory, and in the end, returns the merged histograms.
fn parallel_compression_round_helper<'a, EP, E, C, G, Factory>(
    graph: &G,
    compression_parameters: &CompressionParameters,
    factory: &Factory,
    num_threads: usize,
    msg: impl AsRef<str>,
    cpl: &mut ConcurrentWrapper,
) -> Result<IntegerHistograms<EP>>
where
    EP: EncodeParams + Send + Sync,
    E: Encode + Send + Sync,
    C: ContextModel + Default + Copy + Send + Sync,
    G: SequentialGraph + for<'b> SplitLabeling<SplitLender<'b>: ExactSizeLender + Send>,
    Factory: ThreadEstimatorFactory<'a, E> + Send + Sync,
{
    let num_symbols = 1 << compression_parameters.max_bits;
    let split_iter = graph
        .split_iter(num_threads)
        .into_iter()
        .collect::<Vec<_>>();

    cpl.start(msg);

    // iterate the splitted version of the graph in parallel
    let thread_histograms: Vec<IntegerHistograms<EP>> = split_iter
        .into_iter()
        .enumerate()
        .par_bridge()
        .map_with(
            cpl.clone(),
            |pl, (thread_id, mut thread_lender)| -> Result<IntegerHistograms<EP>> {
                pl.info(format_args!(
                    "Started compression with thread {}",
                    thread_id
                ));

                let Some((node_id, successors)) = thread_lender.next() else {
                    return Err(anyhow::anyhow!(
                        "Empty chunked size of compressors in thread {}",
                        thread_id
                    ));
                };

                let first_node = node_id;

                // Initialize local builder with the estimator from factory
                let mut thread_builder = HuffmanGraphEncoderBuilder::<_, _, EP>::new(
                    num_symbols,
                    factory.create_estimator(),
                    C::default(),
                );
                let offsets_writer = OffsetsWriter::from_write(io::empty(), false)?;

                match compression_parameters.compressor {
                    CompressorType::Approximated { chunk_size } => {
                        let mut compressor = BvCompZ::new(
                            &mut thread_builder,
                            offsets_writer,
                            compression_parameters.compression_window,
                            chunk_size,
                            compression_parameters.max_ref_count,
                            compression_parameters.min_interval_length,
                            first_node,
                        );
                        compressor.push(successors).unwrap();
                        pl.update();
                        for_![ (_, succ) in thread_lender {
                            compressor.push(succ)?;
                            pl.update();
                        }];
                        compressor.flush()?;
                    }
                    CompressorType::Greedy => {
                        let mut compressor = BvComp::new(
                            &mut thread_builder,
                            offsets_writer,
                            compression_parameters.compression_window,
                            compression_parameters.max_ref_count,
                            compression_parameters.min_interval_length,
                            first_node,
                        );
                        compressor.push(successors).unwrap();
                        pl.update();
                        for_![ (_, succ) in thread_lender {
                            compressor.push(succ)?;
                            pl.update();
                        }];
                        compressor.flush()?;
                    }
                }

                Ok(thread_builder.histograms())
            },
        )
        .collect::<Result<Vec<_>>>()?;

    cpl.info(format_args!("Merging histograms from separate threads"));

    // Merge Phase: Combine all local histograms into one
    let mut shared_histograms = IntegerHistograms::<EP>::new(C::num_contexts(), num_symbols);
    for h in thread_histograms {
        shared_histograms.add_all(&h);
    }
    cpl.done();

    Ok(shared_histograms)
}

/// Run the first compression round in parallel using the given estimator type.
/// Splits the graph, compresses each chunk with its own builder, then merges histograms.
fn parallel_first_reference_selection_round<
    EP: EncodeParams + Send + Sync,
    E: Encode + Default + Send + Sync,
    C: ContextModel + Default + Copy + Send + Sync,
    G: SequentialGraph + for<'a> SplitLabeling<SplitLender<'a>: ExactSizeLender + Send>,
>(
    graph: &G,
    compression_parameters: &CompressionParameters,
    num_threads: usize,
    pl: &mut ConcurrentWrapper,
    msg: impl AsRef<str>,
) -> Result<HuffmanGraphEncoderBuilder<E, C, EP>> {
    let factory = DefaultEstimatorFactory::<E>::default();
    let shared_histograms = parallel_compression_round_helper::<EP, E, C, G, _>(
        graph,
        compression_parameters,
        &factory,
        num_threads,
        msg,
        pl,
    )?;

    let builder = HuffmanGraphEncoderBuilder::<_, _, EP>::from_histograms(
        shared_histograms,
        E::default(),
        C::default(),
    );
    Ok(builder)
}

#[allow(clippy::too_many_arguments)]
/// Run one reference-selection pass: collect symbols with the given estimator.
/// Returns a builder seeded with frequency estimates for the next stage.
fn parallel_reference_selection_round<
    EP: EncodeParams + Send + Sync,
    E: Encode,
    C: ContextModel + Default + Copy + Send + Sync,
>(
    graph: &(impl SequentialGraph + for<'a> SplitLabeling<SplitLender<'a>: ExactSizeLender + Send>),
    huffman_graph_encoder_builder: HuffmanGraphEncoderBuilder<E, C, EP>,
    compression_parameters: &CompressionParameters,
    num_threads: usize,
    msg: impl AsRef<str>,
    pl: &mut ConcurrentWrapper,
) -> Result<HuffmanEstimatedEncoderBuilder<EP, C>> {
    // obtain cost model of the previous iteration
    let cost_model = huffman_graph_encoder_builder.histograms().cost();

    // Run parallel compression with Huffman estimator factory
    let factory = HuffmanEstimatorFactory::<'_, EP, C> {
        cost_model: &cost_model,
        _marker: std::marker::PhantomData,
    };
    let shared_histograms = parallel_compression_round_helper::<EP, _, C, _, _>(
        graph,
        compression_parameters,
        &factory,
        num_threads,
        msg,
        pl,
    )?;

    // Finalize builder with Huffman estimator and merged histograms
    let huffman_estimator = HuffmanEstimator::new(cost_model, C::default());
    let builder = HuffmanGraphEncoderBuilder::<_, _, EP>::from_histograms(
        shared_histograms,
        huffman_estimator,
        C::default(),
    );

    Ok(builder)
}

/// Read a BVGraph from `basename` and convert it to a Huffman-encoded graph.
/// The converted graph is written to `output_basename`.
pub fn sequential_convert_graph_file<C: ContextModel + Default + Copy + Send + Sync>(
    basename: impl AsRef<Path>,
    output_basename: impl AsRef<Path>,
    compression_parameters: &CompressionParameters,
) -> Result<()> {
    convert_graph_file::<C>(basename, output_basename, compression_parameters, 1)
}

/// Read a BVGraph from `basename` and convert it to a Huffman-encoded graph running the estimation rounds in parallel.
/// The converted graph is written to `output_basename`.
pub fn parallel_convert_graph_file<C: ContextModel + Default + Copy + Send + Sync>(
    basename: impl AsRef<Path>,
    output_basename: impl AsRef<Path>,
    compression_parameters: &CompressionParameters,
    num_threads: usize,
) -> Result<()> {
    convert_graph_file::<C>(
        basename,
        output_basename,
        compression_parameters,
        num_threads,
    )
}

/// Read a BVGraph from `basename` and convert it to a Huffman-encoded graph.
/// The converted graph is written to `output_basename`.
/// The `parallel` arguments controls if the estimation rounds are executed in parallel or sequentially.
pub fn convert_graph_file<C: ContextModel + Default + Copy + Send + Sync>(
    basename: impl AsRef<Path>,
    output_basename: impl AsRef<Path>,
    compression_parameters: &CompressionParameters,
    num_threads: usize,
) -> Result<()> {
    if basename.as_ref().with_extension(EF_EXTENSION).exists() {
        let graph = BvGraph::with_basename(&basename)
            .endianness::<BE>()
            .load()?;

        convert_graph::<C, _>(&graph, output_basename, compression_parameters, num_threads)
    } else {
        let seq_graph = BvGraphSeq::with_basename(&basename)
            .endianness::<BE>()
            .load()?;

        convert_graph::<C, _>(
            &seq_graph,
            output_basename,
            compression_parameters,
            num_threads,
        )
    }
}

#[allow(clippy::too_many_arguments)]
/// Run the iterative estimation process, starting from the first compression round.
/// Creates the initial builder with the given estimator type and handles both sequential
/// and parallel processing for all rounds.
fn run_conversion_rounds<
    EP: EncodeParams + Send + Sync,
    E: Encode + Default + Send + Sync,
    C: ContextModel + Default + Copy + Send + Sync,
    G: SequentialGraph + for<'a> SplitLabeling<SplitLender<'a>: ExactSizeLender + Send>,
>(
    seq_graph: &G,
    output_basename: impl AsRef<Path>,
    compression_parameters: &CompressionParameters,
    num_threads: usize,
    starting_estimator_name: &str,
    pl: &mut ConcurrentWrapper,
) -> Result<()> {
    let num_symbols = 1 << compression_parameters.max_bits;

    pl.item_name("node")
        .expected_updates(Some(seq_graph.num_nodes()));
    pl.start(format!(
        "Pushing symbols into encoder builder with {}...",
        starting_estimator_name
    ));

    // Run compression for the first round (sequential or parallel)
    let huffman_graph_encoder_builder = if num_threads > 1 {
        parallel_first_reference_selection_round::<EP, E, C, _>(
            seq_graph,
            compression_parameters,
            num_threads,
            pl,
            "",
        )?
    } else {
        let mut builder =
            HuffmanGraphEncoderBuilder::<_, _, EP>::new(num_symbols, E::default(), C::default());
        let offsets_writer = OffsetsWriter::from_write(io::empty(), false)?;
        match compression_parameters.compressor {
            CompressorType::Approximated { chunk_size } => {
                let mut compressor = BvCompZ::new(
                    &mut builder,
                    offsets_writer,
                    compression_parameters.compression_window,
                    chunk_size,
                    compression_parameters.max_ref_count,
                    compression_parameters.min_interval_length,
                    0,
                );
                for_![ (_, succ) in seq_graph.iter() {
                    compressor.push(succ)?;
                    pl.update();
                }];
                compressor.flush()?;
            }
            CompressorType::Greedy => {
                let mut compressor = BvComp::new(
                    &mut builder,
                    offsets_writer,
                    compression_parameters.compression_window,
                    compression_parameters.max_ref_count,
                    compression_parameters.min_interval_length,
                    0,
                );
                for_![ (_, succ) in seq_graph.iter() {
                    compressor.push(succ)?;
                    pl.update();
                }];
                compressor.flush()?;
            }
        }
        builder
    };
    pl.done();

    if compression_parameters.num_rounds == 1 {
        return if num_threads > 1 {
            parallel_write_graph_to_disk(
                &output_basename,
                huffman_graph_encoder_builder,
                seq_graph,
                compression_parameters,
                num_threads,
                pl,
            )
        } else {
            write_graph_to_disk(
                &output_basename,
                huffman_graph_encoder_builder,
                seq_graph,
                compression_parameters,
                pl,
            )
        };
    }

    // second round build the graph with the first Huffman estimator
    let mut huffman_graph_encoder_builder = if num_threads > 1 {
        parallel_reference_selection_round(
            seq_graph,
            huffman_graph_encoder_builder,
            compression_parameters,
            num_threads,
            "Pushing symbols into encoder builder on first round with Huffman estimator...",
            pl,
        )?
    } else {
        reference_selection_round(
            seq_graph,
            huffman_graph_encoder_builder,
            compression_parameters,
            "Pushing symbols into encoder builder on first round with Huffman estimator...",
            pl,
        )?
    };

    // execute all the subsequence rounds
    for round in 2..compression_parameters.num_rounds {
        huffman_graph_encoder_builder = if num_threads > 1 {
            parallel_reference_selection_round(
                seq_graph,
                huffman_graph_encoder_builder,
                compression_parameters,
                num_threads,
                format!(
                    "Pushing symbols into encoder builder with Huffman estimator for round {}...",
                    round + 1
                )
                .as_str(),
                pl,
            )?
        } else {
            reference_selection_round(
                seq_graph,
                huffman_graph_encoder_builder,
                compression_parameters,
                format!(
                    "Pushing symbols into encoder builder with Huffman estimator for round {}...",
                    round + 1
                )
                .as_str(),
                pl,
            )?
        };
    }

    if num_threads > 1 {
        parallel_write_graph_to_disk(
            &output_basename,
            huffman_graph_encoder_builder,
            seq_graph,
            compression_parameters,
            num_threads,
            pl,
        )?;
    } else {
        write_graph_to_disk(
            &output_basename,
            huffman_graph_encoder_builder,
            seq_graph,
            compression_parameters,
            pl,
        )?;
    }

    Ok(())
}

/// Convert a sequential graph to Huffman-encoded form and save to disk.
/// Runs estimation rounds, builds the encoder and writes the compressed graph with its offsets.
pub fn convert_graph<
    C: ContextModel + Default + Copy + Send + Sync,
    G: SequentialGraph + for<'a> SplitLabeling<SplitLender<'a>: ExactSizeLender + Send>,
>(
    seq_graph: &G,
    output_basename: impl AsRef<Path>,
    compression_parameters: &CompressionParameters,
    num_threads: usize,
) -> Result<()> {
    assert!(
        compression_parameters.num_rounds >= 1,
        "Needed at least one estimation round to compress the graph using a Huffman-based encoding."
    );
    let mut pl = concurrent_progress_logger!(
        display_memory = true,
        item_name = "node",
        local_speed = true,
        expected_updates = Some(seq_graph.num_nodes()),
        // log every five minutes
        log_interval = Duration::from_secs(5 * 60),
    );

    match compression_parameters.starting_estimator {
        Estimator::Log2 => {
            run_conversion_rounds::<DefaultEncodeParams, Log2Estimator, C, _>(
                seq_graph,
                &output_basename,
                compression_parameters,
                num_threads,
                "Log2Estimator",
                &mut pl,
            )?;
        }
        Estimator::Fixed => {
            run_conversion_rounds::<DefaultEncodeParams, FixedEstimator, C, _>(
                seq_graph,
                &output_basename,
                compression_parameters,
                num_threads,
                "FixedEstimator",
                &mut pl,
            )?;
        }
    }

    Ok(())
}

/// Parallel version of write_graph_to_disk that compresses the graph in parallel.
/// Each thread compresses a chunk of the graph to temporary files, then the files
/// are concatenated in order to produce the final output.
#[allow(clippy::too_many_arguments)]
fn parallel_write_graph_to_disk<
    EP: EncodeParams + Send + Sync,
    E: Encode,
    C: ContextModel + Default + Copy + Send + Sync,
    G: SequentialGraph + for<'a> SplitLabeling<SplitLender<'a>: ExactSizeLender + Send>,
>(
    output_basename: impl AsRef<Path>,
    huffman_graph_encoder_builder: HuffmanGraphEncoderBuilder<E, C, EP>,
    seq_graph: &G,
    compression_parameters: &CompressionParameters,
    num_threads: usize,
    pl: &mut ConcurrentWrapper,
) -> Result<()> {
    pl.start("Building the encoder with the cost model obtained from estimation rounds...");

    // Create temporary directory for chunk files
    let tmp_dir = tempfile::tempdir().context("Failed to create temporary directory")?;
    let tmp_path = tmp_dir.path();

    // Get histograms and compute cost model for creating per-thread estimators
    let histograms = huffman_graph_encoder_builder.histograms();
    let cost_model = histograms.cost();

    // Build the HuffmanEncoder from the histograms
    let huffman_encoder = HuffmanEncoder::<EP>::new(histograms, compression_parameters.max_bits);

    // Create factory for thread-local HuffmanEstimators
    let estimator_factory = HuffmanEstimatorFactory::<'_, EP, C> {
        cost_model: &cost_model,
        _marker: std::marker::PhantomData,
    };

    // Create output file and write header directly
    // We will keep this writer open and copy chunk bits to it
    let output_path = output_basename.as_ref().with_extension(GRAPH_EXTENSION);
    let outfile = File::create(&output_path)
        .with_context(|| format!("Could not create {}", output_path.display()))?;
    let graph_writer =
        BufBitWriter::<LE, _>::new(WordAdapter::<u32, _>::new(BufWriter::new(outfile)));
    let mut graph_writer = CountBitWriter::<LE, _>::new(graph_writer);

    pl.done();

    pl.info(format_args!("Writing header for the graph..."));
    let header_size = huffman_encoder
        .write_header(&mut graph_writer)
        .context("Failed to write Huffman encoder header")?;

    pl.item_name("node")
        .expected_updates(Some(seq_graph.num_nodes()));
    pl.start("Compressing the graph in parallel...");

    // Split the graph into chunks for parallel processing
    let split_iter: Vec<_> = seq_graph.split_iter(num_threads).into_iter().collect();

    // Prepare work items sequentially (to create estimators from factory which borrows cost_model)
    let work_items: Vec<_> = split_iter
        .into_iter()
        .enumerate()
        .map(|(thread_id, thread_lender)| ChunkedCompressionJob {
            thread_id,
            thread_lender,
            encoder: huffman_encoder.clone(),
            chunk_dir: tmp_path.join(format!("{:016x}", thread_id)),
            compression_params: compression_parameters.clone(),
            estimator: estimator_factory.create_estimator(),
        })
        .collect();

    // Process chunks in parallel using into_par_iter
    let mut chunk_results = work_items
        .into_par_iter()
        .map_with(pl.clone(), |thread_pl, job| {
            let ChunkedCompressionJob {
                thread_id,
                mut thread_lender,
                encoder,
                chunk_dir,
                compression_params,
                estimator,
            } = job;
            // Create chunk directory
            std::fs::create_dir_all(&chunk_dir)
                .with_context(|| format!("Could not create {}", chunk_dir.display()))?;

            let Some((node_id, successors)) = thread_lender.next() else {
                return Err(anyhow::anyhow!("Empty chunk in thread {}", thread_id));
            };

            let first_node = node_id;

            // Create graph and offsets files for this chunk
            let chunk_graph_path = chunk_dir.join("chunk.graph");
            let chunk_offsets_path = chunk_dir.join("chunk.offsets");

            let chunk_file = File::create(&chunk_graph_path)
                .with_context(|| format!("Could not create {}", chunk_graph_path.display()))?;
            let chunk_writer =
                BufBitWriter::<LE, _>::new(WordAdapter::<u32, _>::new(BufWriter::new(chunk_file)));
            let mut chunk_writer = CountBitWriter::<LE, _>::new(chunk_writer);

            // Create encoder for this thread using the HuffmanEstimator
            // constructed from the cost model of the previous estimation rounds
            let chunk_encoder =
                HuffmanGraphEncoder::new(encoder, estimator, C::default(), &mut chunk_writer);

            // Create offsets writer for this chunk
            let chunk_offsets_writer = OffsetsWriter::from_path(&chunk_offsets_path, false)?;

            let stats;
            let mut last_node = first_node;

            match compression_params.compressor {
                CompressorType::Approximated { chunk_size } => {
                    let mut compressor = BvCompZ::new(
                        chunk_encoder,
                        chunk_offsets_writer,
                        compression_params.compression_window,
                        chunk_size,
                        compression_params.max_ref_count,
                        compression_params.min_interval_length,
                        first_node,
                    );
                    compressor.push(successors)?;
                    thread_pl.update();
                    for_![(node, succ) in thread_lender {
                        last_node = node;
                        compressor.push(succ)?;
                        thread_pl.update();
                    }];
                    stats = compressor.flush()?;
                }
                CompressorType::Greedy => {
                    let mut compressor = BvComp::new(
                        chunk_encoder,
                        chunk_offsets_writer,
                        compression_params.compression_window,
                        compression_params.max_ref_count,
                        compression_params.min_interval_length,
                        first_node,
                    );
                    compressor.push(successors)?;
                    thread_pl.update();
                    for_![(node, succ) in thread_lender {
                        last_node = node;
                        compressor.push(succ)?;
                        thread_pl.update();
                    }];
                    stats = compressor.flush()?;
                }
            }

            Ok(ChunkCompressionResult {
                thread_id,
                first_node,
                last_node,
                chunk_dir,
                graph_written_bits: stats.written_bits,
                offsets_written_bits: stats.offsets_written_bits,
                num_arcs: stats.num_arcs,
            })
        })
        .collect::<Result<Vec<_>>>()?;
    chunk_results.sort_by_key(|r| r.thread_id);

    pl.done();

    pl.start("Concatenating compressed chunks to final graph...");

    let header_bits = graph_writer.bits_written as u64;

    // Create offsets file
    let offsets_path = output_basename.as_ref().with_extension(OFFSETS_EXTENSION);
    let mut offsets_writer =
        BufBitWriter::<BE, _>::new(WordAdapter::<usize, _>::new(BufWriter::new(
            File::create(&offsets_path)
                .with_context(|| format!("Could not create {}", offsets_path.display()))?,
        )));
    // Write initial offset (0, gamma-encoded)
    offsets_writer.write_gamma(0)?;

    let mut total_graph_bits = header_bits;
    let mut _total_offsets_bits: u64 = 0;
    let mut _total_arcs: u64 = 0;
    let mut next_expected_node = 0;

    // Concatenate chunks in order, deleting temp directories as we go
    for chunk in chunk_results {
        anyhow::ensure!(
            chunk.first_node == next_expected_node,
            "Non-adjacent chunks: chunk {} has first node {} instead of {}",
            chunk.thread_id,
            chunk.first_node,
            next_expected_node
        );
        next_expected_node = chunk.last_node + 1;

        // Copy graph bits
        let chunk_graph_path = chunk.chunk_dir.join("chunk.graph");
        let chunk_file = File::open(&chunk_graph_path)
            .with_context(|| format!("Could not open {}", chunk_graph_path.display()))?;
        let mut chunk_reader =
            BufBitReader::<LE, _>::new(WordAdapter::<u32, _>::new(BufReader::new(chunk_file)));
        graph_writer
            .copy_from(&mut chunk_reader, chunk.graph_written_bits)
            .with_context(|| {
                format!(
                    "Could not copy graph bits from {}",
                    chunk_graph_path.display()
                )
            })?;
        total_graph_bits += chunk.graph_written_bits;

        // Copy offsets bits
        let chunk_offsets_path = chunk.chunk_dir.join("chunk.offsets");
        let chunk_offsets_file = File::open(&chunk_offsets_path)
            .with_context(|| format!("Could not open {}", chunk_offsets_path.display()))?;
        let mut chunk_offsets_reader = BufBitReader::<BE, _>::new(WordAdapter::<u32, _>::new(
            BufReader::new(chunk_offsets_file),
        ));
        offsets_writer
            .copy_from(&mut chunk_offsets_reader, chunk.offsets_written_bits)
            .with_context(|| {
                format!(
                    "Could not copy offsets bits from {}",
                    chunk_offsets_path.display()
                )
            })?;
        _total_offsets_bits += chunk.offsets_written_bits;
        _total_arcs += chunk.num_arcs;

        // Delete the chunk directory immediately to save disk space
        std::fs::remove_dir_all(&chunk.chunk_dir)
            .with_context(|| format!("Could not remove {}", chunk.chunk_dir.display()))?;

        pl.info(format_args!(
            "Copied chunk {} ({} graph bits, {} offset bits)",
            chunk.thread_id, chunk.graph_written_bits, chunk.offsets_written_bits
        ));
    }

    // Flush writers
    BitWrite::flush(&mut graph_writer)?;
    BitWrite::flush(&mut offsets_writer)?;

    pl.done();

    pl.info(format_args!(
        "After last round with Huffman estimator: Recompressed graph using {} bits ({} bits of header)",
        total_graph_bits, header_size
    ));

    // Write properties file
    let properties = compression_parameters
        .to_properties(
            seq_graph.num_nodes(),
            seq_graph
                .num_arcs_hint()
                .expect("Cannot know how many arcs the source graph contains"),
            total_graph_bits as _,
            C::NAME,
        )
        .context("Cannot serialize properties file")?;
    let properties_path = output_basename.as_ref().with_extension("properties");
    std::fs::write(&properties_path, properties)
        .with_context(|| format!("Could not write {}", properties_path.display()))?;

    // Clean up temp directory (should already be empty)
    drop(tmp_dir);

    Ok(())
}

/// Write the final compressed graph to disk sequentially.
/// Builds the encoder from histograms, writes header, compresses nodes, and saves properties.
fn write_graph_to_disk<
    EP: EncodeParams,
    E: Encode,
    C: ContextModel,
    G: SequentialGraph + for<'a> SplitLabeling<SplitLender<'a>: ExactSizeLender + Send>,
>(
    output_basename: impl AsRef<Path>,
    huffman_graph_encoder_builder: HuffmanGraphEncoderBuilder<E, C, EP>,
    seq_graph: &G,
    compression_parameters: &CompressionParameters,
    pl: &mut ConcurrentWrapper,
) -> Result<()> {
    pl.start("Building the encoder with the cost model obtained from estimation rounds...");

    let output_path = output_basename.as_ref().with_extension(GRAPH_EXTENSION);
    let outfile = File::create(output_path)?;
    let writer = BufBitWriter::<LE, _>::new(WordAdapter::<u32, _>::new(BufWriter::new(outfile)));
    let mut writer = CountBitWriter::<LE, _>::new(writer);
    let mut huffman_graph_encoder =
        huffman_graph_encoder_builder.build(&mut writer, compression_parameters.max_bits);

    pl.done();

    pl.info(format_args!("Writing header for the graph..."));
    let header_size = huffman_graph_encoder.write_header()?;
    let offsets_writer = OffsetsWriter::from_path(
        output_basename.as_ref().with_extension(OFFSETS_EXTENSION),
        false,
    )?;

    pl.item_name("node")
        .expected_updates(Some(seq_graph.num_nodes()));
    pl.start("Compressing the graph...");

    match compression_parameters.compressor {
        CompressorType::Approximated { chunk_size } => {
            let mut compressor = BvCompZ::new(
                huffman_graph_encoder,
                offsets_writer,
                compression_parameters.compression_window,
                chunk_size,
                compression_parameters.max_ref_count,
                compression_parameters.min_interval_length,
                0,
            );
            for_![ (_, succ) in seq_graph.iter() {
                compressor.push(succ)?;
                pl.update();
            }];
            compressor.flush()?;
        }
        CompressorType::Greedy => {
            let mut compressor = BvComp::new(
                huffman_graph_encoder,
                offsets_writer,
                compression_parameters.compression_window,
                compression_parameters.max_ref_count,
                compression_parameters.min_interval_length,
                0,
            );
            for_![ (_, succ) in seq_graph.iter() {
                compressor.push(succ)?;
                pl.update();
            }];
            compressor.flush()?;
        }
    }

    pl.info(format_args!(
        "After last round with Huffman estimator: Recompressed graph using {} bits ({} bits of header)",
        writer.bits_written, header_size
    ));

    let properties = compression_parameters
        .to_properties(
            seq_graph.num_nodes(),
            seq_graph
                .num_arcs_hint()
                .expect("Cannot know how many arcs the source graph contains"),
            writer.bits_written as _,
            C::NAME,
        )
        .context("Cannot serialize properties file")?;
    let properties_path = output_basename.as_ref().with_extension("properties");
    std::fs::write(&properties_path, properties)
        .with_context(|| format!("Could not write {}", properties_path.display()))?;

    pl.done();
    Ok(())
}
