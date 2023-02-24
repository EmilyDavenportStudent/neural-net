use crate::vector::Vector;
use std::f32::consts::E;

pub trait Activator {
    fn activation(x: f32) -> f32;
    fn activation_dx(x: f32) -> f32;
    fn activation_vec(v: &Vector) -> Vector {
        Vector::from(v.0.iter().map(|x| Self::activation(*x)).collect())
    }
    fn activation_dx_vec(v: &Vector) -> Vector {
        Vector::from(v.0.iter().map(|x| Self::activation_dx(*x)).collect())
    }
}

pub struct Sigmoid;
impl Activator for Sigmoid {
    fn activation(x: f32) -> f32 {
        1.0 / (1.0 + (E.powf(-x)))
    }

    fn activation_dx(x: f32) -> f32 {
        Sigmoid::activation(x) * (1.0 - Sigmoid::activation(x))
    }
}

pub struct ReLU;
impl Activator for ReLU {
    fn activation(x: f32) -> f32 {
        match x > 0.0 {
            true => x,
            false => 0.0,
        }
    }

    fn activation_dx(x: f32) -> f32 {
        match x > 0.0 {
            true => 1.0,
            false => 0.0,
        }
    }
}
