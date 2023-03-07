use crate::activation::{Activator, FnActivation, ReLU};
use crate::loss::{mean_squared_error, FnLoss};
use crate::matrix::Matrix;
use crate::vector::Vector;
use rand::{rngs::SmallRng, SeedableRng};
use std::iter::Iterator;

struct Network<T: Activator> {
    layers: Vec<Parameters>,
    activation: T,
    loss: FnLoss,
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
    fn new(layers: Vec<Parameters>, activation: T, loss: FnLoss) -> Self {
        Self {
            layers,
            activation,
            loss,
        }
    }

    fn backpropagate(&self, input: &Vector, expected_output: &Vector) -> Deltas {
        let list = feed_forward_recursive(self.layers.iter(), input, ReLU::activation);
        dbg!(list);
        todo!()
    }
}

#[derive(Debug)]
enum List<T> {
    Cons(T, Box<List<T>>),
    Nil,
}

use List::{Cons, Nil};

fn reverse<T>(l: List<T>) -> List<T> {
    reverse_accumulator(l, Nil)
}

fn reverse_accumulator<T>(l: List<T>, acc: List<T>) -> List<T> {
    match l {
        Nil => acc,
        Cons(car, cdr) => reverse_accumulator(*cdr, Cons(car, Box::new(acc))),
    }
}

struct ListIter<'a, T>(&'a List<T>);

impl<'a, T> Iterator for ListIter<'a, T> {
    type Item = &'a T;

    fn next(&mut self) -> Option<Self::Item> {
        match self.0 {
            Cons(car, cdr) => {
                self.0 = &*cdr;
                Some(car)
            }
            Nil => None,
        }
    }
}

#[test]
fn test_iterate() {
    let list = Cons(1, Box::new(Cons(2, Box::new(Nil))));
    let mut iter = ListIter(&list);
    dbg!(iter.next());
    dbg!(iter.next());
    dbg!(iter.next());
}

#[test]
fn test_reverse() {
    let list = Cons(1, Box::new(Cons(2, Box::new(Cons(3, Box::new(Nil))))));
    let rev = reverse(list);
    match rev {
        Cons(car, cdr) => assert_eq!(3, car),
        Nil => panic!(),
    }
}

fn feed_forward_recursive<'a>(
    mut layers: impl Iterator<Item = &'a Parameters>,
    input: &Vector,
    activation: FnActivation,
) -> List<(Vector, Vector)> {
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

fn backward_pass_recursive<'a>(
    mut layers: impl Iterator<Item = &'a Parameters>,
    expected_output: &Vector,
    loss: FnLoss,
    activation_dx: FnActivation,
) -> Deltas {
    todo!()
}

#[test]
fn some_test_fun() {
    some_fun()
}

fn some_fun() {
    let mut rng = SmallRng::seed_from_u64(69);
    let layers = vec![
        Parameters::new_with_rng(&mut rng, 1, 3),
        Parameters::new_with_rng(&mut rng, 3, 2),
    ];
    let net = Network::new(layers, ReLU, mean_squared_error);
    net.backpropagate(&Vector::from(vec![3.4]), &Vector::from(vec![1.0, 2.0]));
}
