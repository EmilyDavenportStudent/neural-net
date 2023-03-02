use crate::vector::Vector;
use std::f32::consts::E;

pub type FnActivation = fn(&Vector) -> Vector;

pub trait Activator {
    fn activation_scalar(x: f32) -> f32;
    fn activation_dx_scalar(x: f32) -> f32;
    fn activation(v: &Vector) -> Vector {
        Vector::from(v.0.iter().map(|x| Self::activation_scalar(*x)).collect())
    }
    fn activation_dx(v: &Vector) -> Vector {
        Vector::from(v.0.iter().map(|x| Self::activation_dx_scalar(*x)).collect())
    }
}

fn as_vec_operation(scalar_activation: fn(f32) -> f32) -> impl Fn(&Vector) -> Vector {
    move |v: &Vector| Vector::from(v.0.iter().map(|x: &f32| scalar_activation(*x)).collect())
}

pub struct Sigmoid;
impl Activator for Sigmoid {
    fn activation_scalar(x: f32) -> f32 {
        1.0 / (1.0 + (E.powf(-x)))
    }

    fn activation_dx_scalar(x: f32) -> f32 {
        Sigmoid::activation_scalar(x) * (1.0 - Sigmoid::activation_scalar(x))
    }
}

pub struct ReLU;
impl Activator for ReLU {
    fn activation_scalar(x: f32) -> f32 {
        match x > 0.0 {
            true => x,
            false => 0.0,
        }
    }

    fn activation_dx_scalar(x: f32) -> f32 {
        match x > 0.0 {
            true => 1.0,
            false => 0.0,
        }
    }
}
