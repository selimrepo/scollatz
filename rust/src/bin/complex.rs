use num_complex::Complex64;
use std::env;
use std::process;

#[derive(Debug)]
struct Config {
    word: Option<String>,
    seed: Option<u128>,
    real: f64,
    imag: f64,
    repeats: usize,
    max_steps: usize,
}

fn usage() {
    println!("scollatz-complex --word 10110 [--real X --imag Y --repeats N]");
    println!("scollatz-complex --seed N [--real X --imag Y --repeats N]");
    println!("Uses the complex affine extensions E(z)=z/2 and O(z)=(3z+1)/2.");
    println!("A symbolic parity word is not a parity rule for arbitrary complex z.");
}

fn parse() -> Result<Config, String> {
    let mut c = Config {
        word: None, seed: None, real: 1.0, imag: 0.0, repeats: 1, max_steps: 1_000_000,
    };
    let mut args = env::args().skip(1);
    while let Some(arg) = args.next() {
        let value = || -> Result<String, String> { Err("internal parser error".into()) };
        match arg.as_str() {
            "--help" | "-h" => { usage(); process::exit(0); }
            "--word" => c.word = Some(args.next().ok_or("--word needs a binary word")?),
            "--seed" => c.seed = Some(args.next().ok_or("--seed needs a positive integer")?
                .parse().map_err(|_| "invalid --seed")?),
            "--real" => c.real = args.next().ok_or("--real needs a number")?
                .parse().map_err(|_| "invalid --real")?,
            "--imag" => c.imag = args.next().ok_or("--imag needs a number")?
                .parse().map_err(|_| "invalid --imag")?,
            "--repeats" => c.repeats = args.next().ok_or("--repeats needs an integer")?
                .parse().map_err(|_| "invalid --repeats")?,
            "--max-steps" => c.max_steps = args.next().ok_or("--max-steps needs an integer")?
                .parse().map_err(|_| "invalid --max-steps")?,
            _ => return Err(format!("unknown argument: {arg}")),
        }
        let _ = value;
    }
    if c.word.is_some() == c.seed.is_some() {
        return Err("provide exactly one of --word or --seed".into());
    }
    if !c.real.is_finite() || !c.imag.is_finite() || c.repeats == 0 || c.max_steps == 0 {
        return Err("coordinates must be finite and counts must be positive".into());
    }
    Ok(c)
}

fn word_from_seed(mut n: u128, limit: usize) -> Result<String, String> {
    if n == 0 { return Err("seed must be positive".into()); }
    let mut word = String::new();
    for _ in 0..limit {
        if n == 1 { return Ok(word); }
        if n & 1 == 0 {
            word.push('0');
            n /= 2;
        } else {
            word.push('1');
            n = n.checked_mul(3).and_then(|v| v.checked_add(1))
                .ok_or("u128 overflow while generating seed itinerary")? / 2;
        }
    }
    if n == 1 { Ok(word) } else { Err("seed itinerary reached --max-steps before 1".into()) }
}

#[derive(Clone, Copy, Debug)]
struct Affine {
    // The composition is F(z) = a*z + b.
    a: f64,
    b: f64,
}

fn compose_word(word: &str) -> Result<Affine, String> {
    if word.is_empty() { return Err("parity word must not be empty".into()); }
    let mut f = Affine { a: 1.0, b: 0.0 };
    for bit in word.chars() {
        let (scale, offset) = match bit {
            '0' => (0.5, 0.0),       // E(z) = z/2
            '1' => (1.5, 0.5),       // O(z) = (3z+1)/2
            _ => return Err("parity word may contain only 0 and 1".into()),
        };
        f.a *= scale;
        f.b = scale * f.b + offset;
    }
    Ok(f)
}

fn main() {
    let config = match parse() {
        Ok(c) => c,
        Err(e) => { eprintln!("error: {e}"); usage(); process::exit(2); }
    };
    let word = match config.word {
        Some(w) => w,
        None => match word_from_seed(config.seed.unwrap(), config.max_steps) {
            Ok(w) => w,
            Err(e) => { eprintln!("error: {e}"); process::exit(2); }
        },
    };
    let affine = match compose_word(&word) {
        Ok(f) => f,
        Err(e) => { eprintln!("error: {e}"); process::exit(2); }
    };

    let multiplier = Complex64::new(affine.a, 0.0);
    let fixed = if (1.0 - affine.a).abs() < f64::EPSILON {
        None
    } else {
        Some(Complex64::new(affine.b, 0.0) / (Complex64::new(1.0, 0.0) - multiplier))
    };
    let mut z = Complex64::new(config.real, config.imag);
    let initial = z;
    for _ in 0..config.repeats {
        z = multiplier * z + Complex64::new(affine.b, 0.0);
    }

    let odds = word.bytes().filter(|b| *b == b'1').count();
    println!("parity_word={word}");
    println!("word_length={}", word.len());
    println!("odd_branches={odds}");
    println!("complex_initial={:.15}{:+.15}i", initial.re, initial.im);
    println!("complex_after_repeats={:.15}{:+.15}i", z.re, z.im);
    println!("composition_multiplier={:.15}", affine.a);
    println!("composition_offset={:.15}", affine.b);
    println!("multiplier_modulus={:.15}", multiplier.norm());
    println!("fixed_point={}", fixed.map(|v| format!("{:.15}{:+.15}i", v.re, v.im))
        .unwrap_or_else(|| "undefined (unit multiplier)".into()));
    println!("classification={}", if multiplier.norm() < 1.0 {
        "attracting for this fixed symbolic branch"
    } else if multiplier.norm() > 1.0 {
        "repelling for this fixed symbolic branch"
    } else {
        "neutral for this fixed symbolic branch"
    });
    println!("warning=This is a complex extension of a prescribed parity itinerary, not a proof of the Collatz conjecture.");
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn even_branch_is_attracting_with_fixed_point_zero() {
        let f = compose_word("0").unwrap();
        assert!((f.a - 0.5).abs() < 1e-12);
        assert!(f.b.abs() < 1e-12);
    }

    #[test]
    fn odd_branch_has_fixed_point_minus_one() {
        let f = compose_word("1").unwrap();
        assert!((f.a - 1.5).abs() < 1e-12);
        assert!((f.b - 0.5).abs() < 1e-12);
        let fixed = f.b / (1.0 - f.a);
        assert!((fixed + 1.0).abs() < 1e-12);
    }

    #[test]
    fn seed_itinerary_matches_known_trajectory() {
        assert_eq!(word_from_seed(3, 20).unwrap(), "110");
    }

    #[test]
    fn complex_iteration_preserves_imaginary_linear_part() {
        let f = compose_word("00").unwrap();
        let z = Complex64::new(2.0, 4.0);
        let result = Complex64::new(f.a, 0.0) * z + Complex64::new(f.b, 0.0);
        assert!((result.re - 0.5).abs() < 1e-12);
        assert!((result.im - 1.0).abs() < 1e-12);
    }
}
