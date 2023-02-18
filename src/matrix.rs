use crate::vector::{assert_mostly_eq, Vector};
use rand::{rngs::SmallRng, Rng, SeedableRng};

pub struct Matrix(Vec<Vec<f32>>);

impl Matrix {
    pub fn new_with_rng(rng: &mut SmallRng, rows: usize, columns: usize) -> Self {
        Matrix((0..rows).map(|_| gen_rand_row(rng, columns)).collect())
    }

    pub fn from(a: Vec<Vec<f32>>) -> Self {
        Matrix(a)
    }

    pub fn get(&self, row: usize, column: usize) -> f32 {
        self.0[row][column]
    }

    pub fn times_vector(&self, v: &Vector) -> Vector {
        match self.0[0].len() == v.len() {
            true => Vector::from(self.0.iter().map(|row| v.dot_vec(row)).collect()),
            false => panic!("Expected vector length to match matrix height"),
        }
    }
}

macro_rules! matrix {
    [$([$($val:literal),+])+] => {
        Matrix::from(
            vec![$(vec![$($val),+]),+]
        )
    };
}

fn gen_rand_row(rng: &mut SmallRng, size: usize) -> Vec<f32> {
    (0..size).map(|_| rng.gen_range(-1.0..=1.0)).collect()
}

#[test]
fn test_new_with_rng() {
    let mut rng = SmallRng::seed_from_u64(0);
    let a = Matrix::new_with_rng(&mut rng, 2, 2);
    assert_mostly_eq(-0.105349, a.get(0, 0), 0.005);
    assert_mostly_eq(-0.121719, a.get(0, 1), 0.005);
    assert_mostly_eq(0.9597605, a.get(1, 0), 0.005);
    assert_mostly_eq(-0.075665, a.get(1, 1), 0.005);
}

#[test]
fn test_times_vector() {
    let a = matrix![
        [1., -1., 2.]
        [0., -3., 1.]
    ];
    let v = Vector::from(vec![2., 1., 0.]);
    let result = a.times_vector(&v);
    assert_mostly_eq(1., result[0], 0.005);
    assert_mostly_eq(-3., result[1], 0.005);
}

#[test]
fn test_matrix_macro() {
    let a = matrix![
        [1.0, 2.0]
        [3.0, 4.0]
    ];
    assert_eq!(1.0, a.get(0, 0));
    assert_eq!(2.0, a.get(0, 1));
    assert_eq!(3.0, a.get(1, 0));
    assert_eq!(4.0, a.get(1, 1));
}
