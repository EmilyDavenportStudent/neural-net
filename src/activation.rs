use crate::vector::Vector;

trait Activator {
    fn activation(x: f32) -> f32;
    fn activation_dx(x: f32) -> f32;
    fn activation_vec(v: &Vector) -> Vector {
        Vector::from(v.0.iter().map(|x| Self::activation(*x)).collect())
    }
    fn activation_dx_vec(v: &Vector) -> Vector {
        Vector::from(v.0.iter().map(|x| Self::activation_dx(*x)).collect())
    }
}
