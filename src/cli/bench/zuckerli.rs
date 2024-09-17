/*
 * SPDX-FileCopyrightText: 2023 Inria
 * SPDX-FileCopyrightText: 2023 Tommaso Fontana
 *
 * SPDX-License-Identifier: Apache-2.0 OR LGPL-2.1-or-later
 */

use crate::prelude::*;
use anyhow::Result;
use clap::{ArgMatches, Args, Command, FromArgMatches};
use dsi_bitstream::prelude::*;
use itertools::Itertools;
use lender::*;
use rand::rngs::SmallRng;
use rand::Rng;
use rand::SeedableRng;
use std::fs::File;
use std::hint::black_box;
use std::io::BufReader;
use std::path::PathBuf;
use webgraph_rust::bitstreams::BinaryReader;
use webgraph_rust::huffman_zuckerli::huffman_decoder::HuffmanDecoder;
use webgraph_rust::properties;

use webgraph_rust::webgraph::zuckerli_in::{BVGraph, BVGraphBuilder, NUM_CONTEXTS};
use webgraph_rust::{
    properties::Properties,
    utils::encodings::{GammaCode, Huff, UnaryCode, UniversalCode, ZetaCode},
    ImmutableGraph,
};

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

    /// Static dispatch for speed tests (default BvGraph parameters).
    #[arg(short = 'S', long = "static")]
    pub _static: bool,

    /// Test sequential high-speed offset/degree scanning.
    #[arg(short = 'd', long)]
    pub degrees: bool,

    /// Do not test speed, but check that the sequential and random-access successor lists are the same.
    #[arg(short = 'c', long)]
    pub check: bool,
}

pub fn cli(command: Command) -> Command {
    command.subcommand(CliArgs::augment_args(Command::new(COMMAND_NAME)).display_order(0))
}

pub fn main(submatches: &ArgMatches) -> Result<()> {
    let args = CliArgs::from_arg_matches(submatches)?;

    match get_endianness(&args.src)?.as_str() {
        #[cfg(any(
            feature = "be_bins",
            not(any(feature = "be_bins", feature = "le_bins"))
        ))]
        BE::NAME => match args._static {
            true => bench_webgraph::<BE, Static>(args),
            false => bench_webgraph::<BE, Dynamic>(args),
        },
        #[cfg(any(
            feature = "le_bins",
            not(any(feature = "be_bins", feature = "le_bins"))
        ))]
        LE::NAME => match args._static {
            true => bench_webgraph::<LE, Static>(args),
            false => bench_webgraph::<LE, Dynamic>(args),
        },
        e => panic!("Unknown endianness: {}", e),
    }
}

fn create_zuckerli_graph(source_name: &PathBuf) -> Option<ZuckerliGraph> {
    let properties_path = source_name.with_extension("properties");
    let properties_file = File::open(properties_path);
    // TODO: return result with error or at least panic
    let properties_file = properties_file.unwrap();
    // properties_file.unwrap_or_else(|_| panic!("Could not find {}", properties_path.display()));
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
                .into_iter()
                .count();
        }
        println!(
            "Sequential:{:>20} ns/arc",
            (start.elapsed().as_secs_f64() / c as f64) * 1e9
        );

        assert_eq!(c, graph.num_arcs());
    }
}

fn bench_webgraph<E: Endianness, D: Dispatch>(args: CliArgs) -> Result<()>
where
    for<'a> BufBitReader<E, MemWordReader<u32, &'a [u32]>>: CodeRead<E> + BitSeek,
{
    if args.check {
        let graph = BvGraph::with_basename(&args.src).endianness::<E>().load()?;

        let seq_graph = BvGraphSeq::with_basename(&args.src)
            .endianness::<E>()
            .load()?;

        let mut deg_reader = seq_graph.offset_deg_iter();

        // Check that sequential and random-access interfaces return the same result
        for_![ (node, seq_succ) in seq_graph {
            let succ = graph.successors(node);

            assert_eq!(deg_reader.next_degree()?, seq_succ.len());
            assert_eq!(succ.collect_vec(), seq_succ.collect_vec());
        }];
    } else if args.degrees {
        let seq_graph = BvGraphSeq::with_basename(&args.src)
            .endianness::<E>()
            .load()?;

        for _ in 0..args.repeats {
            let mut deg_reader = seq_graph.offset_deg_iter();

            let mut c: u64 = 0;
            let start = std::time::Instant::now();
            for _ in 0..seq_graph.num_nodes() {
                c += black_box(deg_reader.next_degree()? as u64);
            }
            println!(
                "Degrees Only:{:>20} ns/arc",
                (start.elapsed().as_secs_f64() / c as f64) * 1e9
            );

            assert_eq!(c, seq_graph.num_arcs_hint().unwrap());
        }
    } else {
        // webgraph_rust doesn't support dynamic dispatch
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
