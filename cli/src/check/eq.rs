/*
 * SPDX-FileCopyrightText: 2024 Davide Cologni
 *
 * SPDX-License-Identifier: Apache-2.0 OR LGPL-2.1-or-later
 */

use crate::GlobalArgs;
use anyhow::Result;
use clap::Args;
use dsi_bitstream::prelude::*;
use dsi_progress_logger::prelude::*;
use lender::*;
use std::{path::PathBuf, process::exit};
use webgraph::traits::{labels, EqError, SequentialLabeling};

#[derive(Args, Debug)]
#[command(name = "eq", about = "Checks that two graphs have the same contents, listed in the same order. Useful to check equality when two graph are compressed with different parameters or with different algorithms (think about reference selection).", long_about = None)]
pub struct CliArgs {
    /// The basename of the first graph.
    pub first_basename: PathBuf,
    /// The basename of the second graph.
    pub second_basename: PathBuf,
}

pub fn main(_global_args: GlobalArgs, args: CliArgs) -> Result<()> {
    compare_graphs(args)
}

pub fn compare_graphs(args: CliArgs) -> Result<()> {
    let first_graph =
        webgraph::graphs::bvgraph::sequential::BvGraphSeq::with_basename(&args.first_basename)
            .endianness::<BE>()
            .load()?;
    let second_graph =
        webgraph::graphs::bvgraph::sequential::BvGraphSeq::with_basename(&args.second_basename)
            .endianness::<BE>()
            .load()?;

    let mut pl = ProgressLogger::default();
    pl.display_memory(true)
        .item_name("compare graphs")
        .expected_updates(Some(first_graph.num_nodes()));

    pl.start("Start comparing the graphs...");

    if first_graph.num_nodes() != second_graph.num_nodes() {
        let eq_error = EqError::NumNodes {
            first: first_graph.num_nodes(),
            second: second_graph.num_nodes(),
        };
        eprintln!("{}", eq_error);
        exit(1);
    }
    for_!(((node0, succ0), (node1, succ1)) in first_graph.iter().zip(second_graph.iter()) {
        debug_assert_eq!(node0, node1);
        pl.light_update();
        let mut succ0 = succ0.into_iter().collect::<Vec<_>>();
        let mut succ1 = succ1.into_iter().collect::<Vec<_>>();
        succ0.sort();
        succ1.sort();
        let result = labels::eq_succs(node0, succ0, succ1);
        if let Err(eq_error) = result {
            eprintln!("{}", eq_error);
            exit(1);
        }
    });

    pl.done();
    Ok(())
}
