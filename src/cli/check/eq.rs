/*
 * SPDX-FileCopyrightText: 2023 Inria
 * SPDX-FileCopyrightText: 2023 Tommaso Fontana
 *
 * SPDX-License-Identifier: Apache-2.0 OR LGPL-2.1-or-later
 */

use crate::traits::SequentialLabeling;
use anyhow::Result;
use clap::{ArgMatches, Args, Command, FromArgMatches};
use dsi_bitstream::prelude::*;
use dsi_progress_logger::prelude::*;
use lender::Lender;
use std::path::PathBuf;

pub const COMMAND_NAME: &str = "eq";

#[derive(Args, Debug)]
#[command(about = "Checks that two graphs are equals.", long_about = None)]
pub struct CliArgs {
    /// The basename of the first graph.
    pub first_basename: PathBuf,
    /// The basename of the second graph.
    pub second_basename: PathBuf,
}

pub fn cli(command: Command) -> Command {
    command.subcommand(CliArgs::augment_args(Command::new(COMMAND_NAME)).display_order(0))
}

pub fn main(submatches: &ArgMatches) -> Result<()> {
    compare_graphs(CliArgs::from_arg_matches(submatches)?)
}

pub fn compare_graphs(args: CliArgs) -> Result<()> {
    let first_graph =
        crate::graphs::bvgraph::sequential::BvGraphSeq::with_basename(&args.first_basename)
            .endianness::<BE>()
            .load()?;
    let second_graph =
        crate::graphs::bvgraph::sequential::BvGraphSeq::with_basename(&args.second_basename)
            .endianness::<BE>()
            .load()?;

    let mut pl = ProgressLogger::default();
    pl.display_memory(true)
        .item_name("compare graphs")
        .expected_updates(Some(first_graph.num_nodes()));

    let mut first_iter = first_graph.iter().enumerate();
    let mut second_iter = second_graph.iter();
    while let Some((i, (true_node_id, true_succ))) = first_iter.next() {
        let (node_id, succ) = second_iter.next().unwrap();

        assert_eq!(true_node_id, i);
        assert_eq!(true_node_id, node_id);
        assert_eq!(
            true_succ.into_iter().collect::<Vec<_>>(),
            succ.into_iter().collect::<Vec<_>>(),
            "Different successor lists for node {}",
            node_id
        );
        pl.light_update();
    }

    pl.done();
    Ok(())
}
