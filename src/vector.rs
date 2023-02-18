use rand::{rngs::SmallRng, Rng, SeedableRng};
use std::ops::Index;

pub struct Vector(Vec<f32>);

impl Vector {
    pub fn new_from_rng(rng: &mut SmallRng, size: usize) -> Self {
        let contents: Vec<f32> = (0..size).map(|_| rng.gen_range(-1.0..=1.0)).collect();
        Self(contents)
    }

    pub fn from(v: Vec<f32>) -> Self {
        Self(v)
    }

    pub fn plus(&self, v: &Vector) -> Vector {
        match self.0.len() == v.0.len() {
            true => Vector(self.0.iter().zip(&v.0).map(|(x, y)| x + y).collect()),
            false => panic!("Expected equal length vectors"),
        }
    }

    pub fn dot(&self, v: &Vector) -> f32 {
        dot(&self.0, &v.0)
    }

    pub fn dot_vec(&self, v: &Vec<f32>) -> f32 {
        dot(&self.0, &v)
    }

    pub fn len(&self) -> usize {
        self.0.len()
    }
}

pub fn dot(a: &Vec<f32>, b: &Vec<f32>) -> f32 {
    match a.len() == b.len() {
        true => a.iter().zip(b).fold(0.0, |sum, (x, y)| sum + (x * y)),
        false => panic!("Expected equal length vecs"),
    }
}

#[test]
fn test_new_from_rng() {
    let mut rng = SmallRng::seed_from_u64(0);
    let v = Vector::new_from_rng(&mut rng, 3);
    assert_mostly_eq(-0.10534, v[0], 0.005);
    assert_mostly_eq(-0.12171, v[1], 0.005);
    assert_mostly_eq(0.95976, v[2], 0.005);
}

#[test]
fn test_plus() {
    let a = Vector(vec![3.7, -2.8]);
    let b = Vector(vec![9.8, 1.2]);
    let result = a.plus(&b);
    assert_mostly_eq(13.5, result[0], 0.005);
    assert_mostly_eq(-1.6, result[1], 0.005);
}

#[test]
fn dot_test() {
    let a = Vector(vec![6.3, 4.1]);
    let b = Vector(vec![1.2, 9.8]);
    assert_eq!(47.74, a.dot(&b));
}

impl Index<usize> for Vector {
    type Output = f32;

    fn index(&self, index: usize) -> &Self::Output {
        self.0.get(index).unwrap()
    }
}

pub fn assert_mostly_eq(a: f32, b: f32, epsilon: f32) {
    let delta = (b - a).abs();
    assert!(delta <= epsilon);
}
