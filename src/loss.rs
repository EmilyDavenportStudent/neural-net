use crate::vector::Vector;

pub type FnLoss = fn(&Vector, &Vector) -> f32;

pub fn mean_squared_error(actual_values: &Vector, predicted_values: &Vector) -> f32 {
    actual_values
        .0
        .iter()
        .zip(predicted_values.0.iter())
        .fold(0.0, |sum, (actual, expected)| {
            sum + (expected - actual).powi(2)
        })
}
