use crate::matrix::Matrix;
use crate::vector::Vector;
use rand::{rngs::SmallRng, SeedableRng};

struct Network {
    layers: Vec<Parameters>,
}

struct Parameters {
    weights: Matrix,
    biases: Vector,
}

impl Parameters {
    fn new_with_rng(rng: &mut SmallRng, n0_size: usize, n1_size: usize) -> Self {
        Self {
            weights: Matrix::new_with_rng(rng, n1_size, n0_size),
            biases: Vector::new_with_rng(rng, n1_size),
        }
    }
}

impl Network {
    fn new(layers: Vec<Parameters>) -> Self {
        Self { layers }
    }
}

fn some_fun() {
    let mut rng = SmallRng::seed_from_u64(69);
    let layers = vec![
        Parameters::new_with_rng(&mut rng, 1, 3),
        Parameters::new_with_rng(&mut rng, 3, 1),
    ];
    let net = Network::new(layers);
}
