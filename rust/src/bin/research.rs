use num_bigint::BigInt;
use num_rational::BigRational;
use num_traits::{One, Zero};
use std::{env, process};

#[derive(Clone, Debug)]
struct Affine {
    // T^k(n) = (a*n+b)/2^k on the integers following this parity word.
    a: BigInt,
    b: BigInt,
    k: usize,
    odd_count: usize,
}

impl Affine {
    fn identity() -> Self {
        Self { a: BigInt::one(), b: BigInt::zero(), k: 0, odd_count: 0 }
    }

    fn append(&mut self, bit: u8) {
        let (m, c) = if bit == 0 {
            (BigInt::one(), BigInt::zero())
        } else {
            self.odd_count += 1;
            (BigInt::from(3u8), BigInt::one())
        };
        self.b = m.clone() * &self.b + c * (BigInt::one() << self.k);
        self.a *= m;
        self.k += 1;
    }

    fn multiplier(&self) -> BigRational {
        BigRational::new(self.a.clone(), BigInt::one() << self.k)
    }

    fn apply_to_class_residue(&self, residue: &BigInt) -> BigInt {
        let numerator = &self.a * residue + &self.b;
        let denominator = BigInt::one() << self.k;
        debug_assert!((&numerator % &denominator).is_zero());
        numerator / denominator
    }

    fn fixed_point(&self) -> Option<BigRational> {
        let denominator = (BigInt::one() << self.k) - &self.a;
        if denominator.is_zero() { None }
        else { Some(BigRational::new(self.b.clone(), denominator)) }
    }
}

#[derive(Debug)]
struct ParityClass {
    word: Vec<u8>,
    residue: BigInt,
    modulus: BigInt,
    affine: Affine,
    orbit_at_residue: BigInt,
}

impl ParityClass {
    // Each length-k accelerated parity word determines exactly one residue
    // modulo 2^k. Lift the residue one bit at a time, using the fact that
    // changing n by 2^k changes T^k(n) by the odd number 3^odd_count.
    fn from_word(word: &[u8]) -> Self {
        let mut residue = BigInt::zero();
        let mut affine = Affine::identity();
        for &bit in word {
            let current = affine.apply_to_class_residue(&residue);
            let current_bit = if (&current % 2u8).is_zero() { 0 } else { 1 };
            if current_bit != bit {
                residue += BigInt::one() << affine.k;
            }
            affine.append(bit);
        }
        let modulus = BigInt::one() << word.len();
        let orbit_at_residue = affine.apply_to_class_residue(&residue);
        Self { word: word.to_vec(), residue, modulus, affine, orbit_at_residue }
    }

    // For n = residue + q*2^k, T^k(n)-n is
    // (a-2^k)q + (T^k(residue)-residue). If a<2^k, this gives an
    // exact threshold after which every member of the class decreases
    // over one k-step block.
    fn eventual_descent_threshold(&self) -> Option<(BigInt, BigInt)> {
        let gap = &self.modulus - &self.affine.a;
        if gap <= BigInt::zero() { return None; }
        let delta = &self.orbit_at_residue - &self.residue;
        let mut q_min = if delta <= BigInt::zero() {
            BigInt::zero()
        } else {
            delta / &gap + BigInt::one()
        };
        if self.residue.is_zero() && q_min.is_zero() {
            q_min = BigInt::one(); // n must be positive
        }
        let first_n = &self.residue + &q_min * &self.modulus;
        Some((q_min, first_n))
    }
}

fn accelerated_step(n: &BigInt) -> BigInt {
    if (n % 2u8).is_zero() { n / 2u8 } else { (n * 3u8 + 1u8) / 2u8 }
}

fn verify_word(mut n: BigInt, word: &[u8]) -> bool {
    if n <= BigInt::zero() { return false; }
    for &bit in word {
        let actual = if (&n % 2u8).is_zero() { 0 } else { 1 };
        if actual != bit { return false; }
        n = accelerated_step(&n);
    }
    true
}

fn word_from_mask(mask: u64, len: usize) -> Vec<u8> {
    (0..len).map(|i| ((mask >> i) & 1) as u8).collect()
}

fn parse() -> Result<(usize, usize), String> {
    let mut depth = 12usize;
    let mut show_classes = 0usize;
    let mut args = env::args().skip(1);
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--help" | "-h" => {
                println!("scollatz-research [--depth N] [--show-classes N]");
                println!("Partitions all positive integers into exact accelerated parity residue classes modulo 2^N.");
                println!("For every class, computes an exact affine block map and an eventual-descent threshold when available.");
                process::exit(0);
            }
            "--depth" | "--max-period" => depth = args.next().ok_or("--depth needs N")?
                .parse().map_err(|_| "invalid --depth")?,
            "--show-classes" => show_classes = args.next().ok_or("--show-classes needs N")?
                .parse().map_err(|_| "invalid --show-classes")?,
            _ => return Err(format!("unknown option: {arg}")),
        }
    }
    if depth == 0 || depth > 24 {
        return Err("--depth must be in 1..=24 (there are 2^N residue classes)".into());
    }
    Ok((depth, show_classes))
}

fn main() {
    let (depth, show_classes) = match parse() {
        Ok(v) => v,
        Err(e) => { eprintln!("error: {e}"); process::exit(2); }
    };

    println!("system=positive-integer parity-class and orbit-growth research");
    println!("map=T(n)=n/2 for even n; T(n)=(3n+1)/2 for odd n");
    println!("depth={depth} classes={}", 1u64 << depth);
    println!("coverage=all positive integers are partitioned by n mod 2^depth");
    println!("arithmetic=exact BigInt and BigRational; no floating-point proof decisions");

    let total = 1u64 << depth;
    let mut contracting = 0u64;
    let mut noncontracting = 0u64;
    let mut verified = 0u64;
    let mut shown = 0usize;
    let mut odd_histogram = vec![0u64; depth + 1];
    let mut descent_histogram = vec![0u64; depth + 1];

    for mask in 0..total {
        let word = word_from_mask(mask, depth);
        let class = ParityClass::from_word(&word);
        if class.residue >= class.modulus || !verify_word(class.residue.clone(), &word) {
            eprintln!("internal parity-class verification failed for mask={mask}");
            process::exit(3);
        }
        // Check the lifted class against its affine block formula.
        let sample = &class.residue + &class.modulus * BigInt::from(3u8);
        let mut iterated = sample.clone();
        for _ in 0..depth { iterated = accelerated_step(&iterated); }
        let affine_value = &class.orbit_at_residue
            + &class.affine.a * BigInt::from(3u8);
        if iterated != affine_value {
            eprintln!("internal affine-class verification failed for mask={mask}");
            process::exit(3);
        }
        verified += 1;
        odd_histogram[class.affine.odd_count] += 1;

        if let Some((q_min, first_n)) = class.eventual_descent_threshold() {
            contracting += 1;
            descent_histogram[class.affine.odd_count] += 1;
            if shown < show_classes {
                println!(
                    "class word={} residue={} modulus={} odd_steps={} multiplier={} block_offset={} eventual_descent_q>={} first_positive_n={}",
                    word.iter().map(u8::to_string).collect::<String>(),
                    class.residue, class.modulus, class.affine.odd_count,
                    class.affine.multiplier(), class.orbit_at_residue,
                    q_min, first_n
                );
                shown += 1;
            }
        } else {
            noncontracting += 1;
            if shown < show_classes {
                println!(
                    "class word={} residue={} modulus={} odd_steps={} multiplier={} block_offset={} eventual_descent=not-certified-by-this-block",
                    word.iter().map(u8::to_string).collect::<String>(),
                    class.residue, class.modulus, class.affine.odd_count,
                    class.affine.multiplier(), class.orbit_at_residue
                );
                shown += 1;
            }
        }
    }

    println!("classes_verified={verified}");
    println!("eventually_decreasing_classes={contracting}");
    println!("classes_without_block_contraction={noncontracting}");
    println!("odd_step_count_histogram={odd_histogram:?}");
    println!("contracting_class_odd_step_histogram={descent_histogram:?}");
    println!("theorem=within each reported contracting residue class, every positive member n >= first_positive_n has T^depth(n) < n");
    println!("scope=exact finite-depth partition plus classwise asymptotic block bounds");
    println!("limitation=classes without block contraction are not thereby divergent; other block lengths or residue refinements may still give descent");
    println!("proof_status=research certificates for the stated finite-depth class inequalities; not a proof of the Collatz conjecture");
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn affine_composition_matches_small_orbit() {
        let word = [1, 1, 0]; // 3 -> 5 -> 8 -> 4
        let class = ParityClass::from_word(&word);
        assert_eq!(class.residue, BigInt::from(3u8));
        assert_eq!(class.orbit_at_residue, BigInt::from(4u8));
    }

    #[test]
    fn parity_words_give_distinct_complete_residue_classes() {
        for depth in 1..=8 {
            let mut residues = Vec::new();
            for mask in 0..(1u64 << depth) {
                let word = word_from_mask(mask, depth);
                let class = ParityClass::from_word(&word);
                assert!(verify_word(class.residue.clone(), &word));
                residues.push(class.residue);
            }
            residues.sort();
            residues.dedup();
            assert_eq!(residues.len(), 1usize << depth);
            assert_eq!(residues.first(), Some(&BigInt::zero()));
            assert_eq!(residues.last(), Some(&(BigInt::from((1u64 << depth) - 1))));
        }
    }

    #[test]
    fn class_affine_formula_holds_for_many_positive_members() {
        for depth in 1..=7 {
            for mask in 0..(1u64 << depth) {
                let word = word_from_mask(mask, depth);
                let class = ParityClass::from_word(&word);
                for q in 0..5u8 {
                    let n = &class.residue + &class.modulus * BigInt::from(q);
                    if n.is_zero() { continue; }
                    let mut actual = n.clone();
                    for _ in 0..depth { actual = accelerated_step(&actual); }
                    let expected = &class.orbit_at_residue
                        + &class.affine.a * BigInt::from(q);
                    assert_eq!(actual, expected);
                }
            }
        }
    }

    #[test]
    fn descent_threshold_is_exact_for_all_tested_q() {
        for depth in 1..=8 {
            for mask in 0..(1u64 << depth) {
                let word = word_from_mask(mask, depth);
                let class = ParityClass::from_word(&word);
                if let Some((q_min, _)) = class.eventual_descent_threshold() {
                    for offset in 0..20u8 {
                        let q = &q_min + BigInt::from(offset);
                        let n = &class.residue + &class.modulus * &q;
                        let mut after = n.clone();
                        for _ in 0..depth { after = accelerated_step(&after); }
                        assert!(after < n, "depth={depth}, mask={mask}, q={q}");
                    }
                }
            }
        }
    }

    #[test]
    fn known_one_two_cycle_is_in_the_10_class() {
        let class = ParityClass::from_word(&[1, 0]);
        assert_eq!(class.residue, BigInt::from(1u8));
        assert_eq!(class.modulus, BigInt::from(4u8));
        assert_eq!(class.orbit_at_residue, BigInt::from(1u8));
    }
}
