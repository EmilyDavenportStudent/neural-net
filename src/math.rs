struct Matrix<const N: usize, const M: usize>([[f32; M]; N]);
struct Vector<const N: usize>([f32; N]);

#[test]
fn times_vector_test() {
    let A = Matrix([[1., -1., 2.], [0., -3., 1.]]);
    let v = Vector([2., 1., 0.]);
    let result = A.times_vector(&v);
    assert_mostly_eq(1., result.0[0], 0.005);
    assert_mostly_eq(-3., result.0[1], 0.005);
}

impl<const N: usize, const M: usize> Matrix<N, M> {
    fn new() -> Self {
        /*
            N = 3, M = 2
            [[6 4]
             [5 3]
             [9 1]]

            each row of weights/biases correspond to one neuron
        */
        let m: Vec<[f32; M]> = (0..N).map(|_| [0f32; M]).collect();
        Self(m.try_into().unwrap())
    }

    fn times_vector(&self, v: &Vector<M>) -> Vector<N> {
        let product: Vec<f32> = self.0.iter().map(|row| dot(&Vector(*row), v)).collect();
        Vector(product.try_into().unwrap())
    }
}

#[test]
fn plus_vector_test() {
    let a = Vector([3.7, -2.8]);
    let b = Vector([9.8, 1.2]);
    let result = a.plus_vector(&b);
    assert_mostly_eq(13.5, result.0[0], 0.005);
    assert_mostly_eq(-1.6, result.0[1], 0.005);
}

fn assert_mostly_eq(a: f32, b: f32, epsilon: f32) {
    let delta = (b - a).abs();
    assert!(delta <= epsilon);
}

impl<const N: usize> Vector<N> {
    fn plus_vector(&self, v: &Vector<N>) -> Vector<N> {
        let sum: Vec<f32> = self.0.iter().zip(v.0).map(|(x, y)| x + y).collect();
        Vector(sum.try_into().unwrap())
    }
}

#[test]
fn dot_test() {
    let a = Vector([6.3, 4.1]);
    let b = Vector([1.2, 9.8]);
    assert_eq!(47.74, dot(&a, &b));
}

fn dot<const N: usize>(a: &Vector<N>, b: &Vector<N>) -> f32 {
    a.0.iter().zip(b.0).fold(0.0, |sum, (x, y)| sum + (x * y))
}
