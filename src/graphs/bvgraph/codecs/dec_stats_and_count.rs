/*
 * SPDX-FileCopyrightText: 2023 Inria
 * SPDX-FileCopyrightText: 2023 Sebastiano Vigna
 *
 * SPDX-License-Identifier: Apache-2.0 OR LGPL-2.1-or-later
 */

use crate::prelude::*;
use dsi_bitstream::prelude::*;
use std::sync::Mutex;

#[derive(Default, Debug)]
pub struct CodesStatsWithCount {
    pub stats: CodesStats,
    pub count: u64,
}

impl CodesStatsWithCount {
    /// Update the stats with the lengths of the codes for `n` and return
    /// `n` for convenience.
    pub fn update(&mut self, n: u64) -> u64 {
        self.count += 1;
        self.stats.update(n)
    }

    // Combines additively this stats with another one.
    pub fn add(&mut self, rhs: &Self) {
        self.count += rhs.count;
        self.stats.add(&rhs.stats)
    }
}

/// A struct that keeps track of how much bits each piece would take
/// using different codes for compression.
#[derive(Debug, Default)]
pub struct DecoderStatsAndCount {
    /// The statistics for the outdegrees values
    pub outdegrees: CodesStatsWithCount,
    /// The statistics for the reference_offset values
    pub reference_offsets: CodesStatsWithCount,
    /// The statistics for the block_count values
    pub block_counts: CodesStatsWithCount,
    /// The statistics for the blocks values
    pub blocks: CodesStatsWithCount,
    /// The statistics for the interval_count values
    pub interval_counts: CodesStatsWithCount,
    /// The statistics for the interval_start values
    pub interval_starts: CodesStatsWithCount,
    /// The statistics for the interval_len values
    pub interval_lens: CodesStatsWithCount,
    /// The statistics for the first_residual values
    pub first_residuals: CodesStatsWithCount,
    /// The statistics for the residual values
    pub residuals: CodesStatsWithCount,
}

impl DecoderStatsAndCount {
    fn update(&mut self, rhs: &Self) {
        self.outdegrees.add(&rhs.outdegrees);
        self.reference_offsets.add(&rhs.reference_offsets);
        self.block_counts.add(&rhs.block_counts);
        self.blocks.add(&rhs.blocks);
        self.interval_counts.add(&rhs.interval_counts);
        self.interval_starts.add(&rhs.interval_starts);
        self.interval_lens.add(&rhs.interval_lens);
        self.first_residuals.add(&rhs.first_residuals);
        self.residuals.add(&rhs.residuals);
    }
}

/// A wrapper that keeps track of how much bits each piece would take using
/// different codes for compressions for a [`SequentialDecoderFactory`]
/// implementation and returns the stats.
pub struct StatsAndCountDecoderFactory<F: SequentialDecoderFactory> {
    factory: F,
    glob_stats: Mutex<DecoderStatsAndCount>,
}

impl<F> StatsAndCountDecoderFactory<F>
where
    F: SequentialDecoderFactory,
{
    pub fn new(factory: F) -> Self {
        Self {
            factory,
            glob_stats: Mutex::new(DecoderStatsAndCount::default()),
        }
    }

    /// Consume self and return the stats.
    pub fn stats(self) -> DecoderStatsAndCount {
        self.glob_stats.into_inner().unwrap()
    }
}

impl<F> From<F> for StatsAndCountDecoderFactory<F>
where
    F: SequentialDecoderFactory,
{
    #[inline(always)]
    fn from(value: F) -> Self {
        Self::new(value)
    }
}

impl<F> SequentialDecoderFactory for StatsAndCountDecoderFactory<F>
where
    F: SequentialDecoderFactory,
{
    type Decoder<'a>
        = StatsDecoder<'a, F>
    where
        Self: 'a;

    #[inline(always)]
    fn new_decoder(&self) -> anyhow::Result<Self::Decoder<'_>> {
        Ok(StatsDecoder::new(
            self,
            self.factory.new_decoder()?,
            DecoderStatsAndCount::default(),
        ))
    }
}

/// A wrapper over a generic [`Decode`] that keeps track of how much
/// bits each piece would take using different codes for compressions
pub struct StatsDecoder<'a, F: SequentialDecoderFactory> {
    factory: &'a StatsAndCountDecoderFactory<F>,
    codes_reader: F::Decoder<'a>,
    stats: DecoderStatsAndCount,
}

impl<F: SequentialDecoderFactory> Drop for StatsDecoder<'_, F> {
    fn drop(&mut self) {
        self.factory.glob_stats.lock().unwrap().update(&self.stats);
    }
}

impl<'a, F: SequentialDecoderFactory> StatsDecoder<'a, F> {
    /// Wrap a reader
    #[inline(always)]
    pub fn new(
        factory: &'a StatsAndCountDecoderFactory<F>,
        codes_reader: F::Decoder<'a>,
        stats: DecoderStatsAndCount,
    ) -> Self {
        Self {
            factory,
            codes_reader,
            stats,
        }
    }
}

impl<F: SequentialDecoderFactory> Decode for StatsDecoder<'_, F> {
    #[inline(always)]
    fn read_outdegree(&mut self) -> u64 {
        self.stats
            .outdegrees
            .update(self.codes_reader.read_outdegree())
    }

    #[inline(always)]
    fn read_reference_offset(&mut self) -> u64 {
        self.stats
            .reference_offsets
            .update(self.codes_reader.read_reference_offset())
    }

    #[inline(always)]
    fn read_block_count(&mut self) -> u64 {
        self.stats
            .block_counts
            .update(self.codes_reader.read_block_count())
    }

    #[inline(always)]
    fn read_block(&mut self) -> u64 {
        self.stats.blocks.update(self.codes_reader.read_block())
    }

    #[inline(always)]
    fn read_interval_count(&mut self) -> u64 {
        self.stats
            .interval_counts
            .update(self.codes_reader.read_interval_count())
    }

    #[inline(always)]
    fn read_interval_start(&mut self) -> u64 {
        self.stats
            .interval_starts
            .update(self.codes_reader.read_interval_start())
    }

    #[inline(always)]
    fn read_interval_len(&mut self) -> u64 {
        self.stats
            .interval_lens
            .update(self.codes_reader.read_interval_len())
    }

    #[inline(always)]
    fn read_first_residual(&mut self) -> u64 {
        self.stats
            .first_residuals
            .update(self.codes_reader.read_first_residual())
    }

    #[inline(always)]
    fn read_residual(&mut self) -> u64 {
        self.stats
            .residuals
            .update(self.codes_reader.read_residual())
    }
}
