use rand::rngs::SmallRng;

use crate::math::{Matrix, Vector};

struct Network {
    layout: &'static [usize],
}

struct Parameters<const N: usize, const M: usize> {
    weights: Matrix<N, M>,
    biases: Vector<N>,
}

impl<const N: usize, const M: usize> Parameters<N, M> {
    fn new_with_rng(rng: &mut SmallRng) -> Self {
        Self {
            weights: Matrix::<N, M>::new_with_rng(rng),
            biases: Vector::<N>::new_with_rng(rng),
        }
    }
}

impl Network {
    fn new(layout: &'static [usize]) -> Self {
        let shifted = layout.iter().skip(1);
        let base = layout.iter().take(shifted.len());
        todo!()
        // base.zip(shifted).map(|(a, b)| )
    }
}
