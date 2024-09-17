/*
 * SPDX-FileCopyrightText: 2023 Tommaso Fontana
 *
 * SPDX-License-Identifier: Apache-2.0 OR LGPL-2.1-or-later
 */

use anyhow::Result;
use clap::{ArgMatches, Command};

pub mod bf_visit;
pub mod bvgraph;
pub mod zuckerli;

pub const COMMAND_NAME: &str = "bench";

pub fn cli(command: Command) -> Command {
    let sub_command = Command::new(COMMAND_NAME)
        .about("A few benchmark utilities.")
        .subcommand_required(true)
        .arg_required_else_help(true)
        .allow_external_subcommands(true);
    let sub_command = bvgraph::cli(sub_command);
    let sub_command = zuckerli::cli(sub_command);
    let sub_command = bf_visit::cli(sub_command);
    command.subcommand(sub_command.display_order(0))
}

pub fn main(submatches: &ArgMatches) -> Result<()> {
    match submatches.subcommand() {
        Some((bf_visit::COMMAND_NAME, sub_m)) => bf_visit::main(sub_m),
        Some((zuckerli::COMMAND_NAME, sub_m)) => zuckerli::main(sub_m),
        Some((bvgraph::COMMAND_NAME, sub_m)) => bvgraph::main(sub_m),
        Some((command_name, _)) => {
            eprintln!("Unknown command: {:?}", command_name);
            std::process::exit(1);
        }
        None => {
            eprintln!("No command given for build");
            std::process::exit(1);
        }
    }
}
