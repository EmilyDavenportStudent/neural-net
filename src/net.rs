use crate::activation::FnActivation;
use crate::activation::{Activator, ReLU};
use crate::matrix::Matrix;
use crate::vector::Vector;
use rand::{rngs::SmallRng, SeedableRng};
use std::iter::Iterator;

struct Network<T: Activator> {
    layers: Vec<Parameters>,
    activation: T,
}

struct Parameters {
    pub weights: Matrix,
    pub biases: Vector,
}

struct Deltas {
    weights: Vec<Matrix>,
    biases: Vec<Vector>,
}

impl Parameters {
    fn new_with_rng(rng: &mut SmallRng, n0_size: usize, n1_size: usize) -> Self {
        Self {
            weights: Matrix::new_with_rng(rng, n1_size, n0_size),
            biases: Vector::new_with_rng(rng, n1_size),
        }
    }
}

impl<T> Network<T>
where
    T: crate::activation::Activator,
{
    fn new(layers: Vec<Parameters>, activation: T) -> Self {
        Self { layers, activation }
    }

    fn backpropagate(&self, input: &Vector, expected_output: &Vector) -> Deltas {
        let list = feed_forward_recursive(self.layers.iter(), input, ReLU::activation_vec);
        dbg!(list);
        todo!()
    }

    // ugly function but rust doesn't have good generator function syntax yet
    fn feed_all_forward(&self, input: &Vector) -> (Vec<Vector>, Vec<Vector>) {
        let mut outputs = Vec::<(Vector, Vector)>::with_capacity(self.layers.len());
        let mut prev_activation = input.clone();
        for layer in &self.layers {
            let output = Self::feed_forward(&layer, &prev_activation);
            prev_activation = output.0.clone();
            outputs.push(output);
        }
        outputs.into_iter().unzip()
    }

    fn feed_forward(p: &Parameters, input: &Vector) -> (Vector, Vector) {
        let a = p.weights.times_vector(input).plus(&p.biases);
        let z = <T as Activator>::activation_vec(&a);
        (a, z)
    }
}

#[derive(Debug)]
enum FeedForwardList {
    Cons((Vector, Vector), Box<FeedForwardList>),
    Nil,
}

use FeedForwardList::{Cons, Nil};

fn feed_forward_recursive<'a>(
    mut layers: impl Iterator<Item = &'a Parameters>,
    input: &Vector,
    activation: FnActivation,
) -> FeedForwardList {
    match layers.next() {
        Some(p) => {
            let a = p.weights.times_vector(input).plus(&p.biases);
            let z: Vector = activation(&a);
            Cons(
                (a, z.clone()),
                Box::new(feed_forward_recursive(layers, &z, activation)),
            )
        }
        None => Nil,
    }
}

#[test]
fn some_test_fun() {
    some_fun()
}

fn some_fun() {
    let mut rng = SmallRng::seed_from_u64(69);
    let layers = vec![
        Parameters::new_with_rng(&mut rng, 1, 3),
        Parameters::new_with_rng(&mut rng, 3, 1),
    ];
    let net = Network::new(layers, ReLU);
    net.backpropagate(&Vector::from(vec![3.4]), &Vector::from(vec![1.0]));
}
