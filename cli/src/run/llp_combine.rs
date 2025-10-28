/*
 * SPDX-FileCopyrightText: 2025 Inria
 * SPDX-FileCopyrightText: 2025 Sebastiano Vigna
 * SPDX-FileCopyrightText: 2025 Tommaso Fontana
 *
 * SPDX-License-Identifier: Apache-2.0 OR LGPL-2.1-or-later
 */

use crate::run::llp::store_cluster_file;
use crate::{get_thread_pool, GlobalArgs, NumThreadsArg};
use anyhow::Result;
use clap::Parser;
use webgraph_algo::{combine_labels, labels_to_ranks};

use std::path::PathBuf;

#[derive(Parser, Debug)]
#[command(name = "llp-combine", about = "Combine the pre-compute labels from Layered Label Propagation into permutation.", long_about = None)]
pub struct CliArgs {
    /// The folder where the LLP labels are stored.
    pub work_dir: PathBuf,

    /// A filename for the LLP permutation in binary big-endian format.
    pub perm: PathBuf,

    #[arg(short, long)]
    /// Save the permutation in ε-serde format.
    pub epserde: bool,

    /// The number of threads to use.
    #[command(flatten)]
    pub num_threads: NumThreadsArg,
}

pub fn main(_global_args: GlobalArgs, args: CliArgs) -> Result<()> {
    let thread_pool = get_thread_pool(args.num_threads.num_threads);
    thread_pool.install(|| -> Result<()> {
        let labels = combine_labels(args.work_dir)?;
        log::info!("Combined labels...");
        let rank_perm = labels_to_ranks(&labels);
        println!("size of clusters: ");
        let mut cluster_sizes = Vec::new();
        let mut previous_label = labels[0];
        for (i, &x) in rank_perm.iter().enumerate() {
            if labels[x] != previous_label {
                assert!(labels[x] < previous_label);
                previous_label = labels[x];
                cluster_sizes.push(i);
            }
        }
        log::info!("Saving clustering...");
        store_cluster_file(&rank_perm, &cluster_sizes, &args.perm)?;
        Ok(())
    })
}
