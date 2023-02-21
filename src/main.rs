mod matrix;
mod net;
mod vector;
use rand::rngs::SmallRng;
use rand::{Rng, SeedableRng};
use std::f32::consts::E;

fn main() {
    let training_data: Vec<Sample> = vec![
        Sample::from(&[0., 0., 1.], 0.),
        Sample::from(&[0., 1., 1.], 0.),
        Sample::from(&[1., 0., 1.], 1.),
        Sample::from(&[1., 1., 1.], 1.),
    ];
    let mut weights = get_random_weights();
    let mut l1 = 0.0;
    for _n in 0..1000 {
        for sample in &training_data {
            let l0 = sample.input;
            l1 = sigmoid(dot(l0, &weights));
            let l1_error = sample.output - l1;
            let l1_delta = l1_error * dx_sigmoid(l1);
            weights[0] += l0[0] * l1_delta;
            weights[1] += l0[1] * l1_delta;
            weights[2] += l0[2] * l1_delta;
        }
    }
    println!("{}", "Output after training:");
    println!(
        "Sample 1: {}",
        sigmoid(dot(&training_data[0].input, &weights))
    );
    println!(
        "Sample 2: {}",
        sigmoid(dot(&training_data[1].input, &weights))
    );
    println!(
        "Sample 3: {}",
        sigmoid(dot(&training_data[2].input, &weights))
    );
    println!(
        "Sample 4: {}",
        sigmoid(dot(&training_data[3].input, &weights))
    );
    println!(
        "Untrained Sample: {}",
        sigmoid(dot(&[1., 0., 0.], &weights))
    );
}

fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (f32::powf(E, -x)))
}

fn dx_sigmoid(x: f32) -> f32 {
    sigmoid(x) * (1.0 - sigmoid(x))
}

fn get_delta(net_output: f32, expected_output: f32) -> f32 {
    let error = expected_output - net_output;
    error * dx_sigmoid(net_output)
}

fn get_random_weights() -> [f32; 3] {
    let mut rng = SmallRng::seed_from_u64(69u64);
    [
        2. * (rng.gen_range(0.0..1.0) - 1.),
        2. * (rng.gen_range(0.0..1.0) - 1.),
        2. * (rng.gen_range(0.0..1.0) - 1.),
    ]
}

fn dot(a: &[f32], b: &[f32]) -> f32 {
    let mut sum = 0.0;
    for i in 0..a.len() {
        sum += a[i] * b[i];
    }
    sum
}

struct Sample {
    pub input: &'static [f32],
    pub output: f32,
}

impl Sample {
    fn from(input: &'static [f32], output: f32) -> Self {
        Sample { input, output }
    }
}
