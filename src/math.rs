use std::f32::consts::E;

struct Matrix(Vec<Vec<f32>>);
struct Vector(Vec<f32>);

#[test]
fn times_vector_test() {
    let A = Matrix(vec![vec![1., -1., 2.], vec![0., -3., 1.]]);
    let v = Vector(vec![2., 1., 0.]);
    let result = A.times_vector(&v);
    assert_mostly_eq(1., result.0[0], 0.005);
    assert_mostly_eq(-3., result.0[1], 0.005);
}

impl Matrix {
    fn new(rows: i32, columns: i32) -> Self {
        /*
            d = 3,2
            [[6 4]
             [5 3]
             [9 1]]

            each row of weights/biases correspond to one neuron
        */
        Self(
            (0..rows)
                .map(|_| Vec::with_capacity(columns as usize))
                .collect(),
        )
    }

    fn times_vector(&self, v: &Vector) -> Vector {
        let inner = self.0.iter().map(|row| dot(row, &v.0)).collect();
        Vector(inner)
    }
}

#[test]
fn plus_vector_test() {
    let a = Vector(vec![3.7, -2.8]);
    let b = Vector(vec![9.8, 1.2]);
    let result = a.plus_vector(&b);
    assert_mostly_eq(13.5, result.0[0], 0.005);
    assert_mostly_eq(-1.6, result.0[1], 0.005);
}

fn assert_mostly_eq(a: f32, b: f32, epsilon: f32) {
    let delta = (b - a).abs();
    assert!(delta <= epsilon);
}

impl Vector {
    fn plus_vector(&self, v: &Vector) -> Vector {
        let inner = self.0.iter().zip(&v.0).map(|(x, y)| x + y).collect();
        Vector(inner)
    }
}

#[test]
fn dot_test() {
    let a = vec![6.3, 4.1];
    let b = vec![1.2, 9.8];
    assert_eq!(47.74, dot(&a, &b));
}

fn dot(a: &Vec<f32>, b: &Vec<f32>) -> f32 {
    a.iter().zip(b).fold(0.0, |sum, (x, y)| sum + (x * y))
}

#[test]
fn logistic_curve_test() {
    let x = -0.31;
    assert_mostly_eq(0.4231147, logistic_curve(x), 0.005);
}

fn logistic_curve(x: f32) -> f32 {
    1.0 / (1.0 + (E.powf(-x)))
}

#[test]
fn logistic_curve_dx_test() {
    let x = 2.88;
    assert_mostly_eq(0.050326, logistic_curve_dx(x), 0.005);
}

fn logistic_curve_dx(x: f32) -> f32 {
    logistic_curve(x) * (1.0 - logistic_curve(x))
}
