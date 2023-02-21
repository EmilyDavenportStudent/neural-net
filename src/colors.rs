use crate::math::Vector;
use rand::{rngs::SmallRng, SeedableRng};
use std::fmt::Display;

#[test]
fn fmt_vec_test() {
    let mut rng = SmallRng::seed_from_u64(64);
    let v = Vector::<8>::new_with_rng(&mut rng);
    println!("{}", v);
}

const BOX_CHAR: char = '\u{2588}';

impl<const N: usize> Display for Vector<N> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let squares = String::from_iter(self.0.iter().map(|val| value_to_colored_square(*val)));
        let full_string = format!("\x1b[1m[\x1b[22m {} \x1b[1m]\x1b[22m", squares.as_str());
        write!(f, "{}", full_string)
    }
}

fn value_to_colored_square(value: f32) -> String {
    let red = ((value * -1.0).max(0.0) * 255.0).round() as i32;
    let green = (value.max(0.0) * 255.0).round() as i32;
    format!(
        "\x1b[38;2;{};{};{}m{BOX_CHAR}{BOX_CHAR}\x1b[39m",
        red, green, "0"
    )
}
