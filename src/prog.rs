//! Module that contains all of the necessary components for the directed harmonic progression  graph.
use std::collections::{HashMap, HashSet};
use std::iter::FromIterator;
use std::ops::Range;
use twelve_et::prelude::*;
use twelve_et::{
    compute_semi_tone_dist, compute_semi_tone_dist_signed, validate_harmony, PitchClassArithmetic,
    ALTO_VOICE_OCTAVE_RANGE, ALTO_VOICE_PITCH_CLASS_LOWER_BOUND,
    ALTO_VOICE_PITCH_CLASS_UPPER_BOUND, BASS_VOICE_OCTAVE_RANGE,
    BASS_VOICE_PITCH_CLASS_LOWER_BOUND, BASS_VOICE_PITCH_CLASS_UPPER_BOUND, SATB,
    SOPRANO_VOICE_OCTAVE_RANGE, SOPRANO_VOICE_PITCH_CLASS_LOWER_BOUND,
    SOPRANO_VOICE_PITCH_CLASS_UPPER_BOUND, TENOR_VOICE_OCTAVE_RANGE,
    TENOR_VOICE_PITCH_CLASS_LOWER_BOUND, TENOR_VOICE_PITCH_CLASS_UPPER_BOUND,
};

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

#[derive(Debug, Clone, Copy)]
struct PlaceholderSATB((u8, u8), (u8, u8), (u8, u8), (u8, u8));

impl PlaceholderSATB {
    pub fn new(
        soprano: (u8, u8),
        alto: (u8, u8),
        tenor: (u8, u8),
        bass: (u8, u8),
    ) -> PlaceholderSATB {
        PlaceholderSATB(bass, tenor, alto, soprano)
    }
}
/// A helper function for computing the smallest number of semitone needed to get from one pitch to another.
fn find_semi_tone_dist(pitch1: u8, pitch2: u8) -> i32 {
    let dist1 = pitch1.dist(&pitch2);
    let dist2 = pitch2.dist(&pitch1);
    if dist1 <= dist2 {
        dist1 as i32
    } else {
        -(dist2 as i32)
    }
}

/// Helper function for converting semitones from 0 into (pitch_class, octave) format
fn convert_to_pitch_oct(semitones: u32) -> (u8, u8) {
    let oct = semitones / 12;
    let pitch_class = semitones % 12;
    (pitch_class as u8, oct as u8)
}

/// Function that returns the (pitch_class, octave) of the closest pitch with pitch class `new_pitch_class` from `current`. Converts it
/// into (pitch_class, octave) form and returns the 'score' of the transition weighted by `weight`.
fn new_pitch_with_score(
    current: (u8, u8),
    new_pitch_class: u8,
    transition_weight: i32,
) -> ((u8, u8), i32) {
    let dist = match (
        current.0.dist(&new_pitch_class),
        new_pitch_class.dist(&current.0),
    ) {
        (d1, d2) if d1 <= d2 => d1 as i32,
        (_d1, d2) => -(d2 as i32),
    };
    let score = if dist.abs() > 2 {
        transition_weight * ((dist.abs() / 2) + dist.abs() % 2)
    } else {
        0
    };
    let current_semitones = 12 * (current.1 as i32) + (current.0 as i32);
    let new_pitch_semitones = current_semitones + dist;
    let new_pitch = convert_to_pitch_oct(new_pitch_semitones as u32);
    (new_pitch, score)
}

/// A helper function for `find_smoothest_voicing` it will voicings and scores given some current harmony `from` and a set of remaining pitches
/// to choose from that are representatives of the remaining voices in the next harmony `to`.
fn generate_smoothest_voicing(
    current: PlaceholderSATB,
    new_root: u8,
    new_bass: (u8, u8),
    remaining: HashSet<u8>,
) -> Option<Vec<(PlaceholderSATB, i32)>> {
    let mut result: Vec<(PlaceholderSATB, i32)> = vec![];
    for soprano in &remaining {
        let (new_soprano, soprano_score) = new_pitch_with_score(current.3, *soprano, 2);
        for alto in &remaining {
            let (new_alto, alto_score) = new_pitch_with_score(current.2, *alto, 1);
            for tenor in &remaining {
                let (new_tenor, tenor_score) = new_pitch_with_score(current.1, *tenor, 1);
                if validate_harmony(new_root, new_soprano, new_alto, new_tenor, new_bass) {
                    result.push((
                        PlaceholderSATB::new(new_soprano, new_alto, new_tenor, new_bass),
                        soprano_score + alto_score + tenor_score,
                    ));
                }
            }
        }
    }
    if !result.is_empty() {
        return Some(result);
    }
    None
}

/// A function for generating the smoothest voicings from a given harmony `from` to a destination harmony `to`.
/// Function will return a `Option<Vec<(PlaceholderSATB, i32)>>`, Some variant if there is are valid voicing(s), None otherwise.
/// Each valid harmony generated will return with a score that represents how `smooth` the transition is from `from` to `the generated harmony.
/// The function will exclude any voicings that have parallel 5ths/octaves or parallel unison voices.
fn find_smoothest_voicing(
    from: PlaceholderSATB,
    to: &HarmonicProgressionNode,
) -> Option<Vec<(PlaceholderSATB, i32)>> {
    // Find bass notes that have a distance of within 2 semitones from the bass of `from`
    let new_bass_pitches = to
        .pitch_classes
        .iter()
        .filter(|pitch_class| {
            from.0 .0.dist(&pitch_class) <= 2 || pitch_class.dist(&from.0 .0) <= 2
        })
        .map(|pitch_class| *pitch_class)
        .collect::<Vec<u8>>();
    // Check if we have new bass pitches within a distance of 2 semitones from previous bass, if we don't we are done.
    if new_bass_pitches.is_empty() {
        return None;
    }

    // for collecting the resulting voicings if any
    let mut res = vec![];

    // For each new potential bass voice, generate all possible voicings
    for new_bass in new_bass_pitches {
        let mut remaining = to.pitch_classes.clone();
        if (to.root.is_third(&new_bass) && to.root.dist(&new_bass) == 4 && !remaining.contains(&((to.root + 6) % 12)))
            || to.root.is_seventh(&new_bass) {
            remaining.remove(&new_bass);
        }
        let (new_bass_oct, _) = new_pitch_with_score(from.0, new_bass, 0);
        if let Some(ref mut voicings) =
            generate_smoothest_voicing(from, to.root, new_bass_oct, remaining)
        {
            res.append(voicings);
        }
    }

    if !res.is_empty() {
        res.sort_by_key(|(_, score)| *score);
        return Some(res);
    }

    None
}

/// A helper for function for `find_voicings`, performs the depth first search populating the vector `voicings` with valid voicings.
fn find_voicings_dfs(
    voices: &Vec<u8>,
    bounds: &Vec<(Range<u8>, u8, u8)>,
    voicing: &mut Vec<(u8, u8)>,
    voicings: &mut Vec<Vec<(u8, u8)>>,
    i: usize,
) {
    if i == voices.len() {
        voicings.push(voicing.clone());
        return ();
    }
    let cur_voice = voices[i];
    let (cur_octave_range, cur_lower_bound, cur_upper_bound) =
        (bounds[i].0.clone(), bounds[i].1, bounds[i].2);
    let highest_oct = cur_octave_range.len() - 1;
    // Use backtracking to generate the voices
    for (j, oct) in cur_octave_range.enumerate() {
        // TODO: Check the bounds for each voice, ensure that two voices are not greater than an octave apart or twelf apart if considering distance between bass and tenor.
        // Also ensure that adjacent voices do not cross eachother i.e. alto is higher than soprano etc...
        if (j == 0 && cur_voice < cur_lower_bound)
            || (j == highest_oct && cur_voice > cur_upper_bound)
            || (voicing.len() == 1
                && compute_semi_tone_dist(
                    (voicing[voicing.len() - 1].0, voicing[voicing.len() - 1].1),
                    (cur_voice, oct),
                ) > 19)
            || (voicing.len() > 1
                && compute_semi_tone_dist(
                    (voicing[voicing.len() - 1].0, voicing[voicing.len() - 1].1),
                    (cur_voice, oct),
                ) > 12)
            || (voicing.len() >= 1
                && ((voicing[voicing.len() - 1].1 == oct
                    && voicing[voicing.len() - 1].0 > cur_voice)
                    || (voicing[voicing.len() - 1].1 > oct)))
        {
            continue;
        }
        voicing.push((cur_voice, oct));
        find_voicings_dfs(voices, bounds, voicing, voicings, i + 1);
        voicing.pop();
    }
}

/// A funciton for determining valid voicings for the harmony generated by the voices `soprano`, `alto`, `tenor` and bass.
/// Note this function only generates voicings that ensure the voices are within valid ranges, it does not check that the voicings are valid SATB voicings.
/// The function essentially performs a depth first seartch to generate all voicings.
fn find_voicings(soprano: u8, alto: u8, tenor: u8, bass: u8) -> Vec<Vec<(u8, u8)>> {
    let voices = vec![bass, tenor, alto, soprano];
    let bounds = vec![
        (
            BASS_VOICE_OCTAVE_RANGE.clone(),
            BASS_VOICE_PITCH_CLASS_LOWER_BOUND,
            BASS_VOICE_PITCH_CLASS_UPPER_BOUND,
        ),
        (
            TENOR_VOICE_OCTAVE_RANGE.clone(),
            TENOR_VOICE_PITCH_CLASS_LOWER_BOUND,
            TENOR_VOICE_PITCH_CLASS_UPPER_BOUND,
        ),
        (
            ALTO_VOICE_OCTAVE_RANGE.clone(),
            ALTO_VOICE_PITCH_CLASS_LOWER_BOUND,
            ALTO_VOICE_PITCH_CLASS_UPPER_BOUND,
        ),
        (
            SOPRANO_VOICE_OCTAVE_RANGE.clone(),
            SOPRANO_VOICE_PITCH_CLASS_LOWER_BOUND,
            SOPRANO_VOICE_PITCH_CLASS_UPPER_BOUND,
        ),
    ];
    let mut voicing = vec![];
    let mut voicings = vec![];
    // Populate voicings using backtracking/dfs
    find_voicings_dfs(&voices, &bounds, &mut voicing, &mut voicings, 0);
    voicings
}

#[cfg(test)]
mod test {
    use std::thread::current;

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

    #[test]
    fn test_find_voicings() {
        // Test good old root position I in c maj using an open voicing
        let voicings = find_voicings(0, 4, 7, 0);
        println!("{:?}", voicings);

        assert_eq!(voicings, vec![vec![(0, 3), (7, 3), (4, 4), (0, 5)]]);

        let voicings = find_voicings(0, 7, 4, 0);
        println!("{:?}", voicings);

        assert_eq!(
            vec![
                vec![(0, 3), (4, 4), (7, 4), (0, 5)],
                vec![(0, 4), (4, 4), (7, 4), (0, 5)]
            ],
            voicings
        );

        let voicings = find_voicings(7, 0, 3, 0);
        println!("{:?}", voicings);
        assert_eq!(vec![vec![(0, 3), (3, 3), (0, 4), (7, 4)]], voicings);

        let voicings = find_voicings(5, 2, 11, 7);
        println!("{:?}", voicings);

        assert_eq!(
            vec![
                vec![(7, 2), (11, 3), (2, 4), (5, 4)],
                vec![(7, 3), (11, 3), (2, 4), (5, 4)]
            ],
            voicings
        );
    }

    #[test]
    fn test_generate_smoothest_voicings() {
        let current_harmony = PlaceholderSATB::new((0, 5), (4, 4), (7, 3), (0, 3));

        // Test to second inversion V_7
        let mut remaining = HashSet::new();
        remaining.insert(7);
        remaining.insert(11);
        remaining.insert(5);
        let next_voicing = generate_smoothest_voicing(current_harmony, 7, (2, 3), remaining);
        println!("{:?}", next_voicing);

        // Test transition to first inversion V_7
        let mut remaining = HashSet::new();
        remaining.insert(2);
        remaining.insert(5);
        remaining.insert(7);
        let next_voicing = generate_smoothest_voicing(current_harmony, 7, (11, 2), remaining);
        println!("{:?}", next_voicing);
    }

    #[test]
    fn test_find_smoothest_voicing() {
        let current_harmony = PlaceholderSATB::new((0, 5), (4, 4), (7, 3), (0, 3));
        let major_V_7 = harm_prog_node!(0x42 as u8; 7; 11, 2, 5);
        let next_voicings = find_smoothest_voicing(current_harmony, &major_V_7);

        println!("{:?}", next_voicings);

        let major_IV = harm_prog_node!(0xa6 as u8; 5; 9, 0);
        let next_voicings = find_smoothest_voicing(current_harmony, &major_IV);

        println!("{:?}", next_voicings);

        // Test from ii to V
        let current_harmony = PlaceholderSATB::new((2, 5), (5, 4), (9, 3), (2, 3));
        let next_voicings = find_smoothest_voicing(current_harmony, &major_V_7);

        println!("{:?}", next_voicings);

        let current_harmony = PlaceholderSATB::new((2, 5), (2, 4), (9, 3), (5, 3));
        let next_voicings = find_smoothest_voicing(current_harmony, &major_V_7);

        println!("{:?}", next_voicings);

        let current_harmony = PlaceholderSATB::new((0, 5), (7, 4), (0, 4), (4, 3));
        let dim_vii = harm_prog_node!(0x22 as u8; 11; 2, 5);
        let next_voicings = find_smoothest_voicing(current_harmony, &dim_vii);

        println!("{:?}", next_voicings);
    }
}
