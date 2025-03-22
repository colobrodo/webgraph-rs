/*
 * SPDX-FileCopyrightText: 2023 Inria
 * SPDX-FileCopyrightText: 2023 Tommaso Fontana
 *
 * SPDX-License-Identifier: Apache-2.0 OR LGPL-2.1-or-later
 */

use crate::cli::create_parent_dir;
use crate::prelude::*;
use anyhow::Result;
use clap::{ArgMatches, Args, Command, FromArgMatches};
use dsi_bitstream::prelude::*;
use dsi_progress_logger::prelude::*;
use lender::Lender;
use std::{
    fs::File,
    io::{BufWriter, Write},
    path::PathBuf,
    time::Duration,
};

pub const COMMAND_NAME: &str = "bin";

#[derive(Args, Debug)]
#[command(about = "Decompresses a BvGraph, to the binary \"bin\" format used by Zuckerli to compress graphs.", long_about = None)]
pub struct CliArgs {
    /// The basename of the source graph.
    pub src: PathBuf,
    /// The output path of the decompressed graph.
    pub dst: PathBuf,
}

// Simple on-disk representation of a graph that can directly mapped into memory
// (allowing reduced memory usage).
// Format description:
// - 8 bytes of fingerprint
// - 4 bytes to represent the number of nodes N
// - N+1 8-byte integers that represent the index of the first edge of the i-th
//   adjacency list. The last of these integers is the total number of edges, M.
// - M 4-byte integers that represent the destination node of each graph edge.
struct BinGraph {
    offsets: Vec<u64>,
    arcs: Vec<u32>,
}

impl BinGraph {
    fn add_list(&mut self, node_index: usize, successors: impl IntoIterator<Item = usize>) {
        assert_eq!(node_index, self.offsets.len());
        let total_arcs = self.arcs.len();
        self.offsets.push(total_arcs as u64);
        self.arcs.extend(successors.into_iter().map(|x| x as u32));
    }

    fn new() -> Self {
        BinGraph {
            offsets: Vec::new(),
            arcs: Vec::new(),
        }
    }

    fn write(&self, path: PathBuf) -> anyhow::Result<()> {
        // create the file if not exists
        let file = File::create(&path)?;
        let mut writer = BufWriter::new(file);
        // Fingerprint of the simple uncompressed graph format: number of bytes to
        // represent the number of edges followed by number of bytes to represent the
        // number of nodes.
        let fingerprint =
            (std::mem::size_of::<u64>() as u64) << 4 | std::mem::size_of::<u32>() as u64;
        // write the fingerprint
        writer.write_all(&fingerprint.to_le_bytes())?;
        // write the total number of nodes
        let number_of_nodes = self.offsets.len() as u32;
        writer.write_all(&number_of_nodes.to_le_bytes())?;
        // adds to the offsets the total number of arcs
        // write the offsets
        for offset in self.offsets.iter() {
            writer.write_all(&offset.to_le_bytes())?;
        }
        let total_arcs = self.arcs.len() as u64;
        writer.write_all(&total_arcs.to_le_bytes())?;
        // write the nodes
        for node in self.arcs.iter() {
            writer.write_all(&node.to_le_bytes())?;
        }
        writer.flush()?;
        Ok(())
    }
}

pub fn cli(command: Command) -> Command {
    command.subcommand(CliArgs::augment_args(Command::new(COMMAND_NAME)).display_order(0))
}

pub fn main(submatches: &ArgMatches) -> Result<()> {
    let start = std::time::Instant::now();
    let args = CliArgs::from_arg_matches(submatches)?;

    create_parent_dir(&args.dst)?;

    let graph = BvGraphSeq::with_basename(&args.src)
        .endianness::<BE>()
        .load()?;

    let mut pl = ProgressLogger::default();
    // log every five minutes
    pl.log_interval(Duration::from_secs(60));

    pl.item_name("node")
        .expected_updates(Some(graph.num_nodes()));

    let mut bin = BinGraph::new();
    let mut iter = graph.iter();
    while let Some((true_node_id, true_succ)) = iter.next() {
        bin.add_list(true_node_id, true_succ);
        pl.update();
    }
    pl.done();
    bin.write(args.dst)?;

    log::info!(
        "The re-compression took {:.3} seconds",
        start.elapsed().as_secs_f64()
    );
    Ok(())
}
