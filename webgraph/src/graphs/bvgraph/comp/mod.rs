/*
 * SPDX-FileCopyrightText: 2023 Tommaso Fontana
 * SPDX-FileCopyrightText: 2023 Sebastiano Vigna
 *
 * SPDX-License-Identifier: Apache-2.0 OR LGPL-2.1-or-later
 */

mod bvcomp;
mod bvcompz;
pub use bvcomp::*;
pub use bvcompz::*;

mod impls;
pub use impls::BvCompBuilder;

mod flags;
pub use flags::*;
