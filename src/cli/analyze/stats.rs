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
use dsi_progress_logger::prelude::*;
use lender::*;
use std::path::PathBuf;

pub const COMMAND_NAME: &str = "stats";

#[derive(Args, Debug)]
#[command(about = "Reads a graph and print the total space used by each component.", long_about = None)]
pub struct CliArgs {
    /// The basename of the graph.
    pub src: PathBuf,
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
        BE::NAME => optimize_codes::<BE>(args),
        #[cfg(any(
            feature = "le_bins",
            not(any(feature = "be_bins", feature = "le_bins"))
        ))]
        LE::NAME => optimize_codes::<LE>(args),
        e => panic!("Unknown endianness: {}", e),
    }
}

pub fn optimize_codes<E: Endianness + 'static>(args: CliArgs) -> Result<()>
where
    for<'a> BufBitReader<E, MemWordReader<u32, &'a [u32]>>: CodeRead<E> + BitSeek,
{
    // TODO!: speed it up by using random access graph if possible
    let graph = BvGraphSeq::with_basename(args.src)
        .endianness::<E>()
        .load()?
        .map_factory(StatsAndCountDecoderFactory::new);

    let mut pl = ProgressLogger::default();
    pl.display_memory(true)
        .item_name("node")
        .expected_updates(Some(graph.num_nodes()));

    pl.start("Scanning...");

    let mut iter = graph.iter();
    while iter.next().is_some() {
        pl.light_update();
    }
    pl.done();

    drop(iter); // This releases the decoder and updates the global stats
    let stats = graph.into_inner().stats();

    macro_rules! impl_best_code {
        ($old_bits:expr, $stats:expr, $($code:ident - $old:expr),*) => {
            println!("{:>17} {:>16} {:>16} {:>16} {:>16} {:>12}",
                "Type", "Bits", "Bytes", "Elements", "Average size", "Perc",
            );
            $(
                $old_bits += $old;
            )*

            $(
                println!("{:>17} {:>16} {:>16} {:>16} {:>16} {:>12}",
                    stringify!($code),
                    $old,
                    $old / 8,
                    $stats.$code.count,
                    format!("{:.3}", $old as f64 / $stats.$code.count as f64),
                    format!("{:.3}%", 100.0 * ($old) as f64 / $old_bits as f64),
                );
            )*
        };
    }

    let mut old_bits = 0;
    impl_best_code!(
        old_bits,
        stats,
        outdegrees - stats.outdegrees.stats.gamma,
        reference_offsets - stats.reference_offsets.stats.unary,
        block_counts - stats.block_counts.stats.gamma,
        blocks - stats.blocks.stats.gamma,
        interval_counts - stats.interval_counts.stats.gamma,
        interval_starts - stats.interval_starts.stats.gamma,
        interval_lens - stats.interval_lens.stats.gamma,
        first_residuals - stats.first_residuals.stats.zeta[2],
        residuals - stats.residuals.stats.zeta[2]
    );

    println!();
    println!(" bit size: {:>16}", old_bits);
    println!(" byte size: {:>16}", normalize(old_bits as f64 / 8.0));
    Ok(())
}

fn normalize(mut value: f64) -> String {
    let mut uom = ' ';
    if value > 1000.0 {
        value /= 1000.0;
        uom = 'K';
    }
    if value > 1000.0 {
        value /= 1000.0;
        uom = 'M';
    }
    if value > 1000.0 {
        value /= 1000.0;
        uom = 'G';
    }
    if value > 1000.0 {
        value /= 1000.0;
        uom = 'T';
    }
    format!("{:.3}{}", value, uom)
}
