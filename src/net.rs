use crate::math::{Matrix, Vector};
use rand::{rngs::SmallRng, SeedableRng};

macro_rules! construct_network {
    ($($layer_size:literal->$next_layer_size:literal),+) => {
        type Layers = ($(Parameters<$layer_size, $next_layer_size>),+);
    };
}

struct Network<L> {
    layers: L,
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

impl<L> Network<L> {
    fn new(layers: L) -> Self {
        Self { layers }
    }
}

fn some_fun() {
    let mut rng = SmallRng::seed_from_u64(69);
    construct_network!(1->3,3->1);
    let layers: Layers = (
        Parameters::<1, 3>::new_with_rng(&mut rng),
        Parameters::<3, 1>::new_with_rng(&mut rng),
    );
    let net = Network::<Layers>::new(layers);
}
