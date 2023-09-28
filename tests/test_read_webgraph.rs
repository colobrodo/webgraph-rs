/*
 * SPDX-FileCopyrightText: 2023 Inria
 * SPDX-FileCopyrightText: 2023 Sebastiano Vigna
 *
 * SPDX-License-Identifier: Apache-2.0 OR LGPL-2.1-or-later
 */

use std::collections::HashMap;

use anyhow::Result;
use dsi_bitstream::prelude::*;
use epserde::prelude::MemCase;
use sux::prelude::*;
use webgraph::prelude::*;

type ReadType = u32;
type BufferType = u64;

const NODES: usize = 325557;
const ARCS: usize = 3216152;

fn get_bvgraph() -> Result<impl RandomAccessGraph> {
    // Read the offsets
    let mut data = std::fs::read("tests/data/cnr-2000.offsets").unwrap();
    // pad with zeros so we can read with ReadType words
    while data.len() % core::mem::size_of::<ReadType>() != 0 {
        data.push(0);
    }
    // we must do this becasue Vec<u8> is not guaranteed to be properly aligned
    let data = data
        .chunks(core::mem::size_of::<ReadType>())
        .map(|chunk| ReadType::from_ne_bytes(chunk.try_into().unwrap()))
        .collect::<Vec<_>>();

    // Read the offsets gammas
    let mut offsets = Vec::with_capacity(NODES);
    let mut reader = BufBitReader::<BE, BufferType, _>::new(MemWordReaderInf::new(&data));
    let mut offset = 0;
    for _ in 0..NODES {
        offset += reader.read_gamma().unwrap() as usize;
        offsets.push(offset);
    }

    let mut builder = EliasFanoBuilder::new(offsets.len(), offset + 1);
    for o in offsets {
        builder.push(o)?;
    }

    let mut data = std::fs::read("tests/data/cnr-2000.graph").unwrap();
    // pad with zeros so we can read with ReadType words
    while data.len() % core::mem::size_of::<ReadType>() != 0 {
        data.push(0);
    }
    // we must do this becasue Vec<u8> is not guaranteed to be properly aligned
    let data = data
        .chunks(core::mem::size_of::<ReadType>())
        .map(|chunk| ReadType::from_ne_bytes(chunk.try_into().unwrap()))
        .collect::<Vec<_>>();

    let cf = CompFlags::from_properties(&HashMap::new()).unwrap();
    let compression_window = cf.compression_window;
    let min_interval_length = cf.min_interval_length;

    let ef = builder.build();
    let ef: webgraph::EF<Vec<usize>> = ef.convert_to().unwrap();

    // create a random access reader
    Ok(BVGraph::new(
        <DynamicCodesReaderBuilder<BE, _>>::new(data, cf).unwrap(),
        MemCase::encase(ef),
        min_interval_length,
        compression_window,
        NODES,
        ARCS,
    ))
}

#[test]
fn test_iter_nodes() -> Result<()> {
    let bvgraph = get_bvgraph()?;

    let mut seen_node_ids = Vec::new();

    // Check that they read the same
    while let Some((node_id, seq_succ)) = bvgraph.iter_nodes().take(100).next() {
        seen_node_ids.push(node_id);
        let rand_succ = bvgraph.successors(node_id).into_iter().collect::<Vec<_>>();
        assert_eq!(rand_succ, seq_succ.into_iter().collect::<Vec<_>>());
    }

    assert_eq!(
        seen_node_ids,
        (0..bvgraph.num_nodes()).take(100).collect::<Vec<_>>()
    );

    Ok(())
}

#[test]
fn test_iter_nodes_from() -> Result<()> {
    let bvgraph = get_bvgraph()?;
    for i in [0, 1, 2, 5, 10, 100] {
        let mut seen_node_ids = Vec::new();
        // Check that they read the same
        while let Some((node_id, seq_succ)) = bvgraph.iter_nodes_from(i).take(100).next() {
            seen_node_ids.push(node_id);
            let rand_succ = bvgraph.successors(node_id).into_iter().collect::<Vec<_>>();
            assert_eq!(rand_succ, seq_succ.into_iter().collect::<Vec<_>>());
        }

        assert_eq!(
            seen_node_ids,
            (i..bvgraph.num_nodes()).take(100).collect::<Vec<_>>()
        );
    }

    Ok(())
}
