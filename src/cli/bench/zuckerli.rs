/*
 * SPDX-FileCopyrightText: 2023 Inria
 * SPDX-FileCopyrightText: 2023 Tommaso Fontana
 *
 * SPDX-License-Identifier: Apache-2.0 OR LGPL-2.1-or-later
 */

use anyhow::Result;
use cap::Cap;
use clap::{ArgMatches, Args, Command, FromArgMatches};
use rand::rngs::SmallRng;
use rand::Rng;
use rand::SeedableRng;
use std::alloc;
use std::fs::File;
use std::hint::black_box;
use std::io::BufReader;
use std::path::PathBuf;
use tempfile::NamedTempFile;

use webgraph_rust::bitstreams::BinaryReader;
use webgraph_rust::huffman_zuckerli::huffman_decoder::HuffmanDecoder;
use webgraph_rust::utils::EncodingType;

use webgraph_rust::webgraph::zuckerli_in::{BVGraph, BVGraphBuilder, NUM_CONTEXTS};
use webgraph_rust::webgraph::zuckerli_out::BVGraphBuilder as OutBVGraphBuilder;
use webgraph_rust::{
    properties::Properties,
    utils::encodings::{GammaCode, Huff, UnaryCode, ZetaCode},
    ImmutableGraph,
};

#[global_allocator]
static ALLOCATOR: Cap<alloc::System> = Cap::new(alloc::System, usize::MAX);

type ZuckerliGraph = BVGraph<
    Huff,
    GammaCode,
    Huff,
    GammaCode,
    UnaryCode,
    Huff,
    Huff,
    // unused for now
    GammaCode,
    GammaCode,
    GammaCode,
    GammaCode,
    UnaryCode,
    GammaCode,
    ZetaCode,
>;

pub const COMMAND_NAME: &str = "zuckerli";

#[derive(Args, Debug)]
#[command(about = "Benchmarks for the Zuckerli implementation.", long_about = None)]
pub struct CliArgs {
    /// The basename of the graph.
    pub src: PathBuf,

    /// Perform a random-access test on this number of randomly selected nodes.
    #[arg(short, long)]
    pub random: Option<usize>,

    /// The number of repeats.
    #[arg(short = 'R', long, default_value = "10")]
    pub repeats: usize,

    /// In random-access tests, test just access to the first successor.
    #[arg(short = 'f', long)]
    pub first: bool,

    /// Test memory usage during graph compression (the src should refere to the basename of a bvgraph with this option enabled).
    #[arg(short = 'm', long)]
    pub memory: bool,
}

pub fn cli(command: Command) -> Command {
    command.subcommand(CliArgs::augment_args(Command::new(COMMAND_NAME)).display_order(0))
}

pub fn main(submatches: &ArgMatches) -> Result<()> {
    let args = CliArgs::from_arg_matches(submatches)?;

    // zuckerli implementation (webgraph_rust) does not support different endianness, neither dynamic dispatch
    if args.memory {
        bench_memory_usage(&args);
    } else {
        match args.random {
            Some(samples) => {
                bench_random(
                    create_zuckerli_graph(&args.src).unwrap(),
                    samples,
                    args.repeats,
                    args.first,
                );
            }
            None => {
                bench_seq(create_zuckerli_graph(&args.src).unwrap(), args.repeats);
            }
        }
    }
    Ok(())
}

fn create_zuckerli_graph(source_name: &PathBuf) -> Option<ZuckerliGraph> {
    let properties_path = source_name.with_extension("properties");
    let properties_file = File::open(properties_path);
    let properties_file = properties_file.expect("Cannot find the .properties file");
    let p = java_properties::read(BufReader::new(properties_file))
        .unwrap_or_else(|_| panic!("Failed parsing the properties file"));

    let props = Properties::from(p);

    let graph = BVGraphBuilder::<
        Huff,
        GammaCode,
        Huff,
        GammaCode,
        UnaryCode,
        Huff,
        Huff,
        // Default encoding
        GammaCode,
        GammaCode,
        GammaCode,
        GammaCode,
        UnaryCode,
        GammaCode,
        ZetaCode,
    >::new()
    .set_in_min_interval_len(props.min_interval_len)
    .set_in_max_ref_count(props.max_ref_count)
    .set_in_window_size(props.window_size)
    .set_in_zeta(props.zeta_k)
    .set_num_nodes(props.nodes)
    .set_num_edges(props.arcs)
    .load_graph(source_name.to_str()?)
    .load_offsets(source_name.to_str()?)
    .load_outdegrees()
    .build();

    Some(graph)
}

fn bench_memory_usage(args: &CliArgs) {
    let properties_path = args.src.with_extension("properties");
    let properties_file = File::open(properties_path);
    let properties_file = properties_file.expect("Cannot find the .properties file");
    let p = java_properties::read(BufReader::new(properties_file))
        .unwrap_or_else(|_| panic!("Failed parsing the properties file"));
    let props = Properties::from(p);

    match (props.block_coding, props.block_count_coding, props.outdegree_coding, props.offset_coding, props.reference_coding, props.interval_coding, props.residual_coding) {
        (EncodingType::GAMMA, EncodingType::GAMMA, EncodingType::GAMMA, EncodingType::GAMMA, EncodingType::UNARY, EncodingType::GAMMA, EncodingType::ZETA) => {},
        _ => panic!("Only the default encoding types sequence (GAMMA, GAMMA, GAMMA, GAMMA, UNARY, GAMMA, ZETA) is supported for Huffman compression")
    };

    let tmp_file = NamedTempFile::new().unwrap();
    let tmp_path = tmp_file.path();

    // keep the same parameters readed from the properities file
    let mut bvgraph = OutBVGraphBuilder::<
        GammaCode,
        GammaCode,
        GammaCode,
        GammaCode,
        UnaryCode,
        GammaCode,
        ZetaCode,
        Huff,
        GammaCode,
        Huff,
        GammaCode,
        UnaryCode,
        Huff,
        Huff,
    >::new()
    .set_in_min_interval_len(props.min_interval_len)
    .set_out_min_interval_len(props.min_interval_len)
    .set_in_max_ref_count(props.max_ref_count)
    .set_out_max_ref_count(props.max_ref_count)
    .set_in_window_size(props.window_size)
    .set_out_window_size(props.window_size)
    .set_in_zeta(props.zeta_k)
    .set_out_zeta(props.zeta_k)
    .set_num_nodes(props.nodes)
    .set_num_edges(props.arcs)
    .load_graph(&args.src.to_str().unwrap())
    .load_offsets(&args.src.to_str().unwrap())
    .load_outdegrees()
    .build();

    let tmp_path = tmp_path.to_str().unwrap();

    let allocated_before_store = ALLOCATOR.total_allocated();

    bvgraph.store(tmp_path).expect("Failed storing the graph");

    let allocated_by_store = ALLOCATOR.total_allocated() - allocated_before_store;
    println!(
        "Allocated a total of {}B to store the graph",
        allocated_by_store
    );
    println!("With a peak of {}B", ALLOCATOR.max_allocated());
}

fn bench_random(graph: ZuckerliGraph, samples: usize, repeats: usize, first: bool) {
    let mut reader = BinaryReader::new(graph.graph_memory.clone());
    let mut huff_decoder = HuffmanDecoder::new();
    huff_decoder.decode_headers(&mut reader, NUM_CONTEXTS);

    // Random-access speed test
    for _ in 0..repeats {
        let mut rng = SmallRng::seed_from_u64(0);
        let mut c: u64 = 0;
        let num_nodes = graph.num_nodes();
        let start = std::time::Instant::now();
        if first {
            for _ in 0..samples {
                black_box(
                    graph
                        .decode_list(
                            rng.gen_range(0..num_nodes),
                            &mut reader,
                            None,
                            &mut [],
                            &mut huff_decoder,
                        )
                        .into_iter()
                        .next()
                        .unwrap_or(0),
                );
            }
        } else {
            for _ in 0..samples {
                c += black_box(
                    graph
                        .decode_list(
                            rng.gen_range(0..num_nodes),
                            &mut reader,
                            None,
                            &mut [],
                            &mut huff_decoder,
                        )
                        .into_iter()
                        .count() as u64,
                );
            }
        }

        println!(
            "{}:    {:>20} ns/arc",
            if first { "First" } else { "Random" },
            (start.elapsed().as_secs_f64() / c as f64) * 1e9
        );
    }
}

fn bench_seq(graph: ZuckerliGraph, repeats: usize) {
    let mut reader = BinaryReader::new(graph.graph_memory.clone());
    let mut huff_decoder = HuffmanDecoder::new();
    huff_decoder.decode_headers(&mut reader, NUM_CONTEXTS);

    for _ in 0..repeats {
        let mut c: usize = 0;

        let start = std::time::Instant::now();
        let mut iter = graph.iter();
        while let Some(node) = iter.next() {
            c += graph
                .decode_list(node, &mut reader, None, &mut [], &mut huff_decoder)
                .len();
        }
        println!(
            "Sequential:{:>20} ns/arc",
            (start.elapsed().as_secs_f64() / c as f64) * 1e9
        );

        assert_eq!(c, graph.num_arcs());
    }
}
