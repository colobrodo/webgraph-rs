/*
 * SPDX-FileCopyrightText: 2023 Inria
 *
 * SPDX-License-Identifier: Apache-2.0 OR LGPL-2.1-or-later
 */

use anyhow::Result;
use clap::Parser;
use dsi_bitstream::prelude::*;
use itertools::Itertools;
use lender::*;
use rand::rngs::SmallRng;
use rand::Rng;
use rand::SeedableRng;
use std::hint::black_box;
use webgraph::prelude::*;

#[derive(Parser, Debug)]
#[command(about = "Benchmarks the Rust BVGraph implementation", long_about = None)]
struct Args {
    /// The basename of the graph.
    basename: String,

    /// Perform a random-access test on this number of randomly chosen nodes.
    #[arg(short, long)]
    random: Option<usize>,

    /// The number of repeats.
    #[arg(short = 'R', long, default_value = "10")]
    repeats: usize,

    /// In random-access test, test just access to the first successor.
    #[arg(short = 'f', long)]
    first: bool,

    /// Test sequential high-speed offset/degree scanning.
    #[arg(short = 'd', long)]
    degrees: bool,

    /// Do not test speed, but check that the sequential and random-access successor lists are the same.
    #[arg(short = 'c', long)]
    check: bool,
}

fn bench_webgraph<E: Endianness + 'static>(args: Args) -> Result<()>
where
    for<'a> BufBitReader<E, MemWordReader<u32, &'a [u32]>>: CodeRead<E> + BitSeek,
{
    if args.check {
        let graph = BVGraph::with_basename(&args.basename)
            .endianness::<E>()
            .load()?;

        let seq_graph = BVGraphSeq::with_basename(&args.basename)
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
        let seq_graph = BVGraphSeq::with_basename(&args.basename)
            .endianness::<E>()
            .load()?;

        for _ in 0..args.repeats {
            let mut deg_reader = seq_graph.offset_deg_iter();

            let mut c: u64 = 0;
            let start = std::time::Instant::now();
            for _ in 0..seq_graph.num_nodes() {
                black_box(c += deg_reader.next_degree()? as u64);
            }
            println!(
                "Degrees Only:{:>20} ns/arc",
                (start.elapsed().as_secs_f64() / c as f64) * 1e9
            );

            assert_eq!(c, seq_graph.num_arcs_hint().unwrap());
        }
    } else if let Some(random) = args.random {
        let graph = BVGraph::with_basename(&args.basename)
            .endianness::<E>()
            .mode::<LoadMmap>()
            .flags(MemoryFlags::TRANSPARENT_HUGE_PAGES | MemoryFlags::RANDOM_ACCESS)
            .load()?;

        // Random-access speed test
        for _ in 0..args.repeats {
            let mut rng = SmallRng::seed_from_u64(0);
            let mut c: u64 = 0;
            let start = std::time::Instant::now();
            if args.first {
                for _ in 0..random {
                    black_box(
                        graph
                            .successors(rng.gen_range(0..graph.num_nodes()))
                            .next()
                            .unwrap_or(0),
                    );
                }
            } else {
                for _ in 0..random {
                    c += black_box(
                        graph
                            .successors(rng.gen_range(0..graph.num_nodes()))
                            .count() as u64,
                    );
                }
            }

            println!(
                "{}:    {:>20} ns/arc",
                if args.first { "First" } else { "Random" },
                (start.elapsed().as_secs_f64() / c as f64) * 1e9
            );
        }
    }
    // Sequential speed testx
    for _ in 0..args.repeats {
        // Create a sequential reader
        let mut c: u64 = 0;

        let seq_graph = BVGraphSeq::with_basename(&args.basename)
            .endianness::<E>()
            .load()?;

        let start = std::time::Instant::now();
        let mut iter = seq_graph.iter();
        while let Some((_, succ)) = iter.next() {
            c += succ.count() as u64;
        }
        println!(
            "Sequential:{:>20} ns/arc",
            (start.elapsed().as_secs_f64() / c as f64) * 1e9
        );

        assert_eq!(c, seq_graph.num_arcs_hint().unwrap());
    }
    Ok(())
}

pub fn main() -> Result<()> {
    let args = Args::parse();

    stderrlog::new()
        .verbosity(2)
        .timestamp(stderrlog::Timestamp::Second)
        .init()?;

    match get_endianess(&args.basename)?.as_str() {
        #[cfg(any(
            feature = "be_bins",
            not(any(feature = "be_bins", feature = "le_bins"))
        ))]
        BE::NAME => bench_webgraph::<BE>(args),
        #[cfg(any(
            feature = "le_bins",
            not(any(feature = "be_bins", feature = "le_bins"))
        ))]
        LE::NAME => bench_webgraph::<LE>(args),
        e => panic!("Unknown endianness: {}", e),
    }
}
