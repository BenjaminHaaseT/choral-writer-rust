//! Module that contains all of the necessary components for the directed harmonic progression  graph.
use std::collections::{HashMap, HashSet};
use std::iter::FromIterator;
use twelve_et::prelude::*;
use twelve_et::SATB;

/// Struct that acts as a single node in the directed graph of harmonic progression.
#[derive(Debug, PartialEq)]
struct HarmonicProgressionNode {
    pitch_classes: HashSet<u8>,
    edges: u8,
}

impl Eq for HarmonicProgressionNode {}

/// General macro for creating a harmoinc progression node, gives a general way to construct a harmonic progression node from a variable number of voices.
///
/// `Panics`
/// If `edges` and `pitch` are not of type `u8`, the macro will panic.
macro_rules! harm_prog_node {
    ($edges:expr; $( $pitch:expr ),*) => {
        {
            let mut pitch_classes = HashSet::new();
            $(pitch_classes.insert($pitch);)*
            HarmonicProgressionNode {pitch_classes, edges: $edges}

        }
    };
}

/// General macro for producing a `HarmonicProgressionGraph`, only requires a `u8` and a `bool`
macro_rules! harm_prog_graph {
    ($key:expr) => {
        harm_prog_graph!($key, true)
    };
    ($key:expr, $major:expr) => {
        match ($key, $major) {
            (k, true) => {
                let root = $key;
                let second = (root + 2) % 12;
                let third = (second + 2) % 12;
                let fourth = (third + 1) % 12;
                let fifth = (fourth + 2) % 12;
                let sixth = (fifth + 2) % 12;
                let seventh = (sixth + 2) % 12;
                let major_I = harm_prog_node!(root; root, third, fifth);
                let minor_ii_7 = harm_prog_node!(second; second, fourth, sixth, root);

            }
            (k, false) => {}
            (_, _) => panic!("unaccepted arguments"),
        }
    };
}

/// The struct that implements the directed graph of harmonic progression for a given key.
pub struct HarmonicProgressionGraph {
    graph: HashMap<u8, HarmonicProgressionNode>,
}

// impl HarmonicProgressionGraph {
//     fn build_major_key(key: u8) -> HashMap<u8, (HashSet<u8>, u8)> {
//         let root = key;
//         let second = (key + 2) % 12;
//         let third = (second + 2) % 12;
//         let fourth = (third + 1) % 12;
//         let fifth = (fourth + 2) % 12;
//         let sixth = (fifth + 2) % 12;
//         let seventh = (sixth + 2) % 12;
//         let major_I = harm_prog_node!(0xef as u8, root, third, fifth);
//     }

//     pub fn new(key: u8, major: bool) -> HarmonicProgressionGraph {}
// }

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn test_harm_prog_node_macro() {
        let major_I = harm_prog_node!(0xef as u8; 0, 4, 7);
        println!("{:?}", major_I);
        assert!(true);
    }
}
