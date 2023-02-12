use crate::math::{Matrix, Vector};
use rand::rngs::SmallRng;

struct Network {
    rng: SmallRng,
    layers: Vec<Box<dyn IsParameters>>,
}

trait IsParameters {
    fn inner(&self) -> &Self
    where
        Self: Sized;
}

impl<const N: usize, const M: usize> IsParameters for Parameters<N, M> {
    fn inner(&self) -> &Self
    where
        Self: Sized,
    {
        self
    }
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
    fn new() -> Self {
        todo!()
    }

    fn add_layer<const N: usize, const M: usize>(&mut self) {
        let p = Parameters::<N, M>::new_with_rng(&mut self.rng);
        let p_box = Box::new(p) as Box<dyn IsParameters>;
        self.layers.push(p_box);
    }
}
