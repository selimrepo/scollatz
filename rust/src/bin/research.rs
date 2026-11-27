use num_bigint::BigInt;
use num_rational::BigRational;
use num_traits::{One, Zero, Signed, ToPrimitive};
use std::{env, process};

#[derive(Clone, Debug)]
struct Affine {
    // F_w(x) = (a*x+b)/2^k, exactly.
    a: BigInt,
    b: BigInt,
    k: usize,
    odd_count: usize,
}

impl Affine {
    fn from_word(word: &[u8]) -> Self {
        let mut a = BigInt::one();
        let mut b = BigInt::zero();
        for (i, bit) in word.iter().enumerate() {
            let (m, c) = if *bit == 0 {
                (BigInt::one(), BigInt::zero())
            } else {
                (BigInt::from(3u8), BigInt::one())
            };
            b = m.clone() * b + c * (BigInt::one() << i);
            a *= m;
        }
        Self { a, b, k: word.len(), odd_count: word.iter().filter(|&&x| x == 1).count() }
    }

    fn multiplier(&self) -> BigRational {
        BigRational::new(self.a.clone(), BigInt::one() << self.k)
    }

    fn fixed_point(&self) -> Option<BigRational> {
        let denominator = (BigInt::one() << self.k) - &self.a;
        if denominator.is_zero() { None }
        else { Some(BigRational::new(self.b.clone(), denominator)) }
    }
}

fn accelerated_step(n: &BigInt) -> BigInt {
    if (n % 2u8).is_zero() { n / 2u8 } else { (n * 3u8 + 1u8) / 2u8 }
}

fn valid_word(mut n: BigInt, word: &[u8]) -> bool {
    if n <= BigInt::zero() { return false; }
    for &bit in word {
        let odd = (&n % 2u8) != BigInt::zero();
        if odd != (bit == 1) { return false; }
        n = accelerated_step(&n);
    }
    n == BigInt::from(1u8) || n > BigInt::zero()
}

fn exact_cycle_candidate(word: &[u8], affine: &Affine) -> Option<BigInt> {
    let fixed = affine.fixed_point()?;
    if !fixed.is_integer() { return None; }
    let n = fixed.to_integer();
    if n <= BigInt::zero() { return None; }
    let mut x = n.clone();
    for _ in word {
        x = accelerated_step(&x);
    }
    if x != n || !valid_word(n.clone(), word) { return None; }
    Some(n)
}

fn parse() -> Result<usize, String> {
    let mut max_period = 16usize;
    let mut args = env::args().skip(1);
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--help" | "-h" => {
                println!("scollatz-research [--max-period N]");
                println!("Enumerates all accelerated parity words up to N and applies exact BigInt/BigRational tests.");
                println!("Complex dynamics are reported through exact affine multipliers and fixed points.");
                process::exit(0);
            }
            "--max-period" => max_period = args.next().ok_or("--max-period needs N")?
                .parse().map_err(|_| "invalid --max-period")?,
            _ => return Err(format!("unknown option: {arg}")),
        }
    }
    if max_period == 0 || max_period > 24 {
        return Err("--max-period must be in 1..=24 (exhaustive search is exponential)".into());
    }
    Ok(max_period)
}

fn word_from_mask(mask: u64, len: usize) -> Vec<u8> {
    (0..len).map(|i| ((mask >> i) & 1) as u8).collect()
}

fn main() {
    let max_period = match parse() {
        Ok(v) => v,
        Err(e) => { eprintln!("error: {e}"); process::exit(2); }
    };
    println!("method=exhaustive exact affine branch enumeration");
    println!("max_period={max_period}");
    println!("arithmetic=BigInt + BigRational (no floating point in cycle decisions)");
    println!("word_convention=0: n/2; 1: (3n+1)/2");
    let mut words_checked = 0u64;
    let mut candidate_count = 0u64;

    for len in 1..=max_period {
        let total = 1u64 << len;
        let mut attracting = 0u64;
        let mut repelling = 0u64;
        let mut neutral = 0u64;
        for mask in 0..total {
            let word = word_from_mask(mask, len);
            let affine = Affine::from_word(&word);
            let multiplier = affine.multiplier();
            match multiplier.cmp(&BigRational::one()) {
                std::cmp::Ordering::Less => attracting += 1,
                std::cmp::Ordering::Greater => repelling += 1,
                std::cmp::Ordering::Equal => neutral += 1,
            }
            words_checked += 1;
            if let Some(n) = exact_cycle_candidate(&word, &affine) {
                candidate_count += 1;
                println!(
                    "exact_period_word={} candidate={} multiplier={} fixed_point={}",
                    word.iter().map(u8::to_string).collect::<String>(),
                    n,
                    multiplier,
                    affine.fixed_point().unwrap()
                );
            }
        }
        println!(
            "period_length={len} words={} attracting={} repelling={} neutral={}",
            total, attracting, repelling, neutral
        );
    }

    println!("words_checked={words_checked}");
    println!("validated_cycle_word_hits={candidate_count}");
    println!("scope=finite exhaustive search only; this is not an exclusion of all longer cycles or divergent trajectories");
    println!("proof_status=no proof of the Collatz conjecture");
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn affine_composition_matches_small_orbit() {
        let w = [1, 1, 0]; // 3 -> 5 -> 8 -> 4
        let f = Affine::from_word(&w);
        let numerator = &f.a * BigInt::from(3u8) + &f.b;
        assert_eq!(numerator / (BigInt::one() << f.k), BigInt::from(4u8));
    }

    #[test]
    fn exact_fixed_point_for_all_odd_word() {
        let f = Affine::from_word(&[1]);
        assert_eq!(f.fixed_point().unwrap(), BigRational::from_integer(BigInt::from(-1)));
    }

    #[test]
    fn multiplier_is_exact_rational() {
        let f = Affine::from_word(&[1, 0]);
        assert_eq!(f.multiplier(), BigRational::new(BigInt::from(3), BigInt::from(4)));
    }

    #[test]
    fn rejects_non_integer_fixed_points() {
        let f = Affine::from_word(&[1, 0]);
        assert!(f.fixed_point().unwrap().is_integer() == false);
        assert!(exact_cycle_candidate(&[1, 0], &f).is_none());
    }
}
