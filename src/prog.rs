//! Module that contains all of the necessary components for the directed harmonic progression  graph.
use std::collections::{HashMap, HashSet};
use std::iter::FromIterator;
use twelve_et::prelude::*;
use twelve_et::SATB;

/// Struct that acts as a single node in the directed graph of harmonic progression.
#[derive(Debug, PartialEq)]
struct HarmonicProgressionNode {
    root: u8,
    pitch_classes: HashSet<u8>,
    edges: HashSet<u8>,
}

impl Eq for HarmonicProgressionNode {}

/// General macro for creating a harmoinc progression node, gives a general way to construct a harmonic progression node from a variable number of voices.
///
/// `Panics`
/// If `edges` and `pitch` are not of type `u8`, the macro will panic.
macro_rules! harm_prog_node {
    ($edges:expr; $root:expr; $( $pitch:expr ),*) => {
        {
            let mut pitch_classes = HashSet::new();
            $(pitch_classes.insert($pitch);)*
            let mut new_edges = HashSet::new();
            for i in 0..8 {
                if $edges & (1 << i) != 0 {
                    new_edges.insert(i);
                }
            }
            pitch_classes.insert($root);
            HarmonicProgressionNode {root: $root, pitch_classes, edges: new_edges}

        }
    };
}

/// The struct that implements the directed graph of harmonic progression for a given key.
#[derive(Debug, PartialEq)]
pub struct HarmonicProgressionGraph {
    graph: HashMap<u8, HarmonicProgressionNode>,
}

/// General macro for producing a `HarmonicProgressionGraph`, only requires a `u8` and a `bool`
macro_rules! harm_prog_graph {
    ($key:expr) => {
        harm_prog_graph!($key, true)
    };
    ($key:expr, $major:expr) => {
        match ($key, $major)  {
            (k, true) if (0..12).contains(&k) => {
                let root = k;
                let second = (root + 2) % 12;
                let third = (second + 2) % 12;
                let fourth = (third + 1) % 12;
                let fifth = (fourth + 2) % 12;
                let sixth = (fifth + 2) % 12;
                let seventh = (sixth + 2) % 12;
                let major_I = harm_prog_node!(0xfc as u8; root; third, fifth);
                let minor_ii_7 = harm_prog_node!(0xa0 as u8; second; fourth, sixth, root);
                let minor_iii = harm_prog_node!(0x50 as u8; third; fifth, seventh);
                let major_IV = harm_prog_node!(0xa6 as u8; fourth; sixth, root);
                let major_V_7 = harm_prog_node!(0x42 as u8; fifth; seventh, second, fourth);
                let minor_vi = harm_prog_node!(0x14 as u8; sixth; root, third);
                let dim_vii = harm_prog_node!(0x22 as u8; seventh; second, fourth);
                let mut graph = HashMap::new();
                graph.insert(1, major_I);
                graph.insert(2, minor_ii_7);
                graph.insert(3, minor_iii);
                graph.insert(4, major_IV);
                graph.insert(5, major_V_7);
                graph.insert(6, minor_vi);
                graph.insert(7, dim_vii);
                HarmonicProgressionGraph { graph }
            }
            (k, false) if (0..12).contains(&k) => {
                let root = k;
                let second = (root + 2) % 12;
                let third = (second + 1) % 12;
                let fourth = (third + 2) % 12;
                let fifth = (fourth + 2) % 12;
                let sixth = (fifth + 1) % 12;
                let seventh = (sixth + 2) % 12;
                let min_i = harm_prog_node!(0xfd as u8; root; third, fifth);
                let dim_ii = harm_prog_node!(0xa0 as u8; second; fourth, sixth);
                let maj_III = harm_prog_node!(0x50 as u8; third; fifth, seventh);
                let min_iv = harm_prog_node!(0xa7 as u8; fourth; sixth, root);
                let maj_V_7 = harm_prog_node!(0x42 as u8; fifth; (seventh + 1) % 12, second, fourth);
                let maj_VI = harm_prog_node!(0x14 as u8; sixth; root, third);
                let dim_vii = harm_prog_node!(0x22 as u8; (seventh + 1) % 12; second, fourth);
                let maj_VII = harm_prog_node!(0x8 as u8; seventh; second, fourth);
                let mut graph = HashMap::new();
                graph.insert(1, min_i);
                graph.insert(2, dim_ii);
                graph.insert(3, maj_III);
                graph.insert(4, min_iv);
                graph.insert(5, maj_V_7);
                graph.insert(6, maj_VI);
                graph.insert(7, dim_vii);
                graph.insert(0, maj_VII);
                HarmonicProgressionGraph {graph}

            },
            (k, _) if !(0..12).contains(&k) => panic!("{k} is an unaccepted key"),
            _ => panic!("unexpected arguments"),
        }
    };
}

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn test_harm_prog_node_macro() {
        // Make sure each node gets created correctly, using key of C.
        let major_I = harm_prog_node!(0xfc as u8; 0; 4, 7);
        let test_node = {
            let mut edges = HashSet::new();
            for i in 2..=7_u8 {
                edges.insert(i);
            }
            let mut pitch_classes = HashSet::new();
            pitch_classes.insert(0);
            pitch_classes.insert(4);
            pitch_classes.insert(7);
            HarmonicProgressionNode {
                root: 0,
                pitch_classes,
                edges,
            }
        };
        println!("{:?}", major_I);
        assert_eq!(major_I, test_node);

        // let major_I = harm_prog_node!(0xfc as u8; root, third, fifth);
        let minor_ii_7 = harm_prog_node!(0xa0 as u8; 2; 5, 9, 0);
        let test_node = {
            let mut edges = HashSet::new();
            edges.insert(5);
            edges.insert(7);
            let mut pitch_classes = HashSet::new();
            pitch_classes.insert(2);
            pitch_classes.insert(5);
            pitch_classes.insert(9);
            pitch_classes.insert(0);
            HarmonicProgressionNode {
                root: 2,
                pitch_classes,
                edges,
            }
        };
        assert_eq!(minor_ii_7, test_node);
        println!("{:?}", minor_ii_7);

        let minor_iii = harm_prog_node!(0x50 as u8; 4; 7, 11);
        let test_node = {
            let mut edges = HashSet::new();
            edges.insert(4);
            edges.insert(6);
            let mut pitch_classes = HashSet::new();
            pitch_classes.insert(4);
            pitch_classes.insert(7);
            pitch_classes.insert(11);
            HarmonicProgressionNode {
                root: 4,
                pitch_classes,
                edges,
            }
        };
        assert_eq!(minor_iii, test_node);
        println!("{:?}", minor_iii);

        let major_IV = harm_prog_node!(0xa6 as u8; 5; 9, 0);
        let test_node = {
            let mut edges = HashSet::new();
            edges.insert(5);
            edges.insert(2);
            edges.insert(7);
            edges.insert(1);
            let mut pitch_classes = HashSet::new();
            pitch_classes.insert(5);
            pitch_classes.insert(9);
            pitch_classes.insert(0);
            HarmonicProgressionNode {
                root: 5,
                pitch_classes,
                edges,
            }
        };
        assert_eq!(major_IV, test_node);
        println!("{:?}", major_IV);

        let major_V_7 = harm_prog_node!(0x42 as u8; 7; 11, 2, 5);
        let test_node = {
            let mut edges = HashSet::new();
            edges.insert(1);
            edges.insert(6);
            let mut pitch_classes = HashSet::new();
            pitch_classes.insert(7);
            pitch_classes.insert(11);
            pitch_classes.insert(2);
            pitch_classes.insert(5);
            HarmonicProgressionNode {
                root: 7,
                pitch_classes,
                edges,
            }
        };
        assert_eq!(major_V_7, test_node);
        println!("{:?}", major_V_7);

        let minor_vi = harm_prog_node!(0x14 as u8; 9; 0, 4);
        let test_node = {
            let mut edges = HashSet::new();
            edges.insert(2);
            edges.insert(4);
            let mut pitch_classes = HashSet::new();
            pitch_classes.insert(9);
            pitch_classes.insert(0);
            pitch_classes.insert(4);
            HarmonicProgressionNode {
                root: 9,
                pitch_classes,
                edges,
            }
        };
        assert_eq!(minor_vi, test_node);
        println!("{:?}", minor_vi);

        let dim_vii = harm_prog_node!(0x22 as u8; 11; 2, 5);
        let mut test_node = {
            let mut edges = HashSet::new();
            edges.insert(1);
            edges.insert(5);
            let mut pitch_classes = HashSet::new();
            pitch_classes.insert(11);
            pitch_classes.insert(2);
            pitch_classes.insert(5);
            HarmonicProgressionNode {
                root: 11,
                pitch_classes,
                edges,
            }
        };
        assert_eq!(dim_vii, test_node);
        println!("{:?}", dim_vii);
    }

    #[test]
    fn test_harm_prog_graph_maj_macro() {
        let key_of_c_maj = harm_prog_graph!(0);
        let test_graph = {
            let maj_I = {
                let mut edges = HashSet::new();
                for i in 2..=7_u8 {
                    edges.insert(i);
                }
                let mut pitch_classes = HashSet::new();
                pitch_classes.insert(0);
                pitch_classes.insert(4);
                pitch_classes.insert(7);
                HarmonicProgressionNode {
                    root: 0,
                    pitch_classes,
                    edges,
                }
            };
            let min_ii = {
                let mut edges = HashSet::new();
                edges.insert(5);
                edges.insert(7);
                let mut pitch_classes = HashSet::new();
                pitch_classes.insert(2);
                pitch_classes.insert(5);
                pitch_classes.insert(9);
                pitch_classes.insert(0);
                HarmonicProgressionNode {
                    root: 2,
                    pitch_classes,
                    edges,
                }
            };
            let min_iii = {
                let mut edges = HashSet::new();
                edges.insert(4);
                edges.insert(6);
                let mut pitch_classes = HashSet::new();
                pitch_classes.insert(4);
                pitch_classes.insert(7);
                pitch_classes.insert(11);
                HarmonicProgressionNode {
                    root: 4,
                    pitch_classes,
                    edges,
                }
            };
            let maj_IV = {
                let mut edges = HashSet::new();
                edges.insert(5);
                edges.insert(2);
                edges.insert(7);
                edges.insert(1);
                let mut pitch_classes = HashSet::new();
                pitch_classes.insert(5);
                pitch_classes.insert(9);
                pitch_classes.insert(0);
                HarmonicProgressionNode {
                    root: 5,
                    pitch_classes,
                    edges,
                }
            };
            let maj_V = {
                let mut edges = HashSet::new();
                edges.insert(1);
                edges.insert(6);
                let mut pitch_classes = HashSet::new();
                pitch_classes.insert(7);
                pitch_classes.insert(11);
                pitch_classes.insert(2);
                pitch_classes.insert(5);
                HarmonicProgressionNode {
                    root: 7,
                    pitch_classes,
                    edges,
                }
            };
            let min_vi = {
                let mut edges = HashSet::new();
                edges.insert(2);
                edges.insert(4);
                let mut pitch_classes = HashSet::new();
                pitch_classes.insert(9);
                pitch_classes.insert(0);
                pitch_classes.insert(4);
                HarmonicProgressionNode {
                    root: 9,
                    pitch_classes,
                    edges,
                }
            };
            let dim_vii = {
                let mut edges = HashSet::new();
                edges.insert(1);
                edges.insert(5);
                let mut pitch_classes = HashSet::new();
                pitch_classes.insert(11);
                pitch_classes.insert(2);
                pitch_classes.insert(5);
                HarmonicProgressionNode {
                    root: 11,
                    pitch_classes,
                    edges,
                }
            };
            let mut graph = HashMap::new();
            graph.insert(1, maj_I);
            graph.insert(2, min_ii);
            graph.insert(3, min_iii);
            graph.insert(4, maj_IV);
            graph.insert(5, maj_V);
            graph.insert(6, min_vi);
            graph.insert(7, dim_vii);
            HarmonicProgressionGraph { graph }
        };

        for i in 1..8 {
            assert_eq!(test_graph.graph[&i], key_of_c_maj.graph[&i]);
        }

        assert_eq!(test_graph, key_of_c_maj);
    }

    #[test]
    fn test_harm_prog_graph_min_macro() {
        let key_of_c_min = harm_prog_graph!(0, false);
        println!("{:#?}", key_of_c_min);

        let test_key = {
            let min_i = {
                let mut edges = HashSet::new();
                for i in 0..8_u8 {
                    if i != 1 {
                        edges.insert(i);
                    }
                }
                let mut pitch_classes = HashSet::new();
                pitch_classes.insert(0);
                pitch_classes.insert(3);
                pitch_classes.insert(7);
                HarmonicProgressionNode {
                    root: 0,
                    pitch_classes,
                    edges,
                }
            };
            let dim_ii = {
                let mut edges = HashSet::new();
                edges.insert(5);
                edges.insert(7);
                let mut pitch_classes = HashSet::new();
                pitch_classes.insert(2);
                pitch_classes.insert(5);
                pitch_classes.insert(8);
                HarmonicProgressionNode {
                    root: 2,
                    pitch_classes,
                    edges,
                }
            };
            let maj_III = {
                let mut edges = HashSet::new();
                edges.insert(6);
                edges.insert(4);
                let mut pitch_classes = HashSet::new();
                pitch_classes.insert(3);
                pitch_classes.insert(7);
                pitch_classes.insert(10);
                HarmonicProgressionNode {
                    root: 3,
                    pitch_classes,
                    edges,
                }
            };
            let min_iv = {
                let mut edges = HashSet::new();
                edges.insert(7);
                edges.insert(5);
                edges.insert(1);
                edges.insert(2);
                edges.insert(0);
                let mut pitch_classes = HashSet::new();
                pitch_classes.insert(5);
                pitch_classes.insert(8);
                pitch_classes.insert(0);
                HarmonicProgressionNode {
                    root: 5,
                    pitch_classes,
                    edges,
                }
            };
            let maj_V_7 = {
                let mut edges = HashSet::new();
                edges.insert(1);
                edges.insert(6);
                let mut pitch_classes = HashSet::new();
                pitch_classes.insert(7);
                pitch_classes.insert(11);
                pitch_classes.insert(2);
                pitch_classes.insert(5);
                HarmonicProgressionNode {
                    root: 7,
                    pitch_classes,
                    edges,
                }
            };
            let maj_VI = {
                let mut edges = HashSet::new();
                edges.insert(2);
                edges.insert(4);
                let mut pitch_classes = HashSet::new();
                pitch_classes.insert(8);
                pitch_classes.insert(0);
                pitch_classes.insert(3);
                HarmonicProgressionNode {
                    root: 8,
                    pitch_classes,
                    edges,
                }
            };
            let dim_vii = {
                let mut edges = HashSet::new();
                edges.insert(5);
                edges.insert(1);
                let mut pitch_classes = HashSet::new();
                pitch_classes.insert(11);
                pitch_classes.insert(2);
                pitch_classes.insert(5);
                HarmonicProgressionNode {
                    root: 11,
                    pitch_classes,
                    edges,
                }
            };
            let maj_VII = {
                let mut edges = HashSet::new();
                edges.insert(3);
                let mut pitch_classes = HashSet::new();
                pitch_classes.insert(10);
                pitch_classes.insert(2);
                pitch_classes.insert(5);
                HarmonicProgressionNode {
                    root: 10,
                    pitch_classes,
                    edges,
                }
            };
            let mut graph = HashMap::new();
            graph.insert(1, min_i);
            graph.insert(2, dim_ii);
            graph.insert(3, maj_III);
            graph.insert(4, min_iv);
            graph.insert(5, maj_V_7);
            graph.insert(6, maj_VI);
            graph.insert(7, dim_vii);
            graph.insert(0, maj_VII);
            HarmonicProgressionGraph { graph }
        };

        for k in 0..8 {
            assert_eq!(test_key.graph[&k], key_of_c_min.graph[&k]);
        }

        assert_eq!(test_key, key_of_c_min);
    }
}
