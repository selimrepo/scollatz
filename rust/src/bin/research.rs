use num_bigint::BigInt;
use num_rational::BigRational;
use num_traits::{One, Zero};
use serde::{Deserialize, Serialize};
use std::{env, fs, process};

#[derive(Clone, Debug)]
struct Affine {
    a: BigInt,
    b: BigInt,
    k: usize,
    odd_count: usize,
}
impl Affine {
    fn identity() -> Self { Self { a: BigInt::one(), b: BigInt::zero(), k: 0, odd_count: 0 } }
    fn append(&mut self, bit: u8) {
        let (m, c) = if bit == 0 { (BigInt::one(), BigInt::zero()) }
            else { self.odd_count += 1; (BigInt::from(3u8), BigInt::one()) };
        self.b = m.clone() * &self.b + c * (BigInt::one() << self.k);
        self.a *= m;
        self.k += 1;
    }
    fn multiplier(&self) -> BigRational {
        BigRational::new(self.a.clone(), BigInt::one() << self.k)
    }
    fn apply(&self, n: &BigInt) -> BigInt {
        let denominator = BigInt::one() << self.k;
        let numerator = &self.a * n + &self.b;
        debug_assert!((&numerator % &denominator).is_zero());
        numerator / denominator
    }
}

#[derive(Debug)]
struct ParityClass {
    residue: BigInt,
    modulus: BigInt,
}
impl ParityClass {
    fn from_word(word: &[u8]) -> Self {
        let mut residue = BigInt::zero();
        let mut affine = Affine::identity();
        for &bit in word {
            let current = affine.apply(&residue);
            let current_bit = if (&current % 2u8).is_zero() { 0 } else { 1 };
            if current_bit != bit { residue += BigInt::one() << affine.k; }
            affine.append(bit);
        }
        Self { residue, modulus: BigInt::one() << word.len() }
    }
}

fn step(n: &BigInt) -> BigInt {
    if (n % 2u8).is_zero() { n / 2u8 } else { (n * 3u8 + 1u8) / 2u8 }
}
fn run_steps(n: &BigInt, count: usize) -> BigInt {
    let mut x = n.clone();
    for _ in 0..count { x = step(&x); }
    x
}
fn verify_word(mut n: BigInt, word: &[u8]) -> bool {
    if n <= BigInt::zero() { return false; }
    for &bit in word {
        let actual = if (&n % 2u8).is_zero() { 0 } else { 1 };
        if actual != bit { return false; }
        n = step(&n);
    }
    true
}
fn word_string(word: &[u8]) -> String {
    word.iter().map(u8::to_string).collect()
}
fn parse_word(s: &str) -> Result<Vec<u8>, String> {
    s.bytes().map(|b| match b { b'0' => Ok(0), b'1' => Ok(1), _ => Err("parity word must contain only 0 and 1".into()) }).collect()
}

#[derive(Clone, Debug, Serialize, Deserialize)]
struct DescentCertificate {
    word: String,
    residue: String,
    modulus: String,
    block_length: usize,
    odd_steps: usize,
    q_min: String,
    first_positive_n: String,
}
#[derive(Clone, Debug, Serialize, Deserialize)]
struct UnresolvedClass {
    word: String,
    residue: String,
    modulus: String,
}
#[derive(Clone, Debug, Serialize, Deserialize)]
struct ProofCertificate {
    format: String,
    max_depth: usize,
    visited_nodes: usize,
    descent_classes: Vec<DescentCertificate>,
    unresolved_classes: Vec<UnresolvedClass>,
    base_verified_up_to: Option<String>,
    complete_induction_certificate: bool,
}

fn find_descent(word: &[u8], residue: &BigInt, depth: usize) -> Option<DescentCertificate> {
    let modulus = BigInt::one() << depth;
    // A deeper class fixes more parity bits. Any prefix block j<=depth
    // remains affine on the entire deeper class.
    for j in 1..=depth {
        let mut affine = Affine::identity();
        for &bit in &word[..j] { affine.append(bit); }
        let offset = affine.apply(residue);
        let slope = affine.a * (BigInt::one() << (depth - j));
        let gap = &modulus - &slope;
        if gap <= BigInt::zero() { continue; }
        let delta = &offset - residue;
        let mut q_min = if delta < BigInt::zero() { BigInt::zero() }
            else { delta / &gap + BigInt::one() };
        if residue.is_zero() && q_min.is_zero() { q_min = BigInt::one(); }
        let first = residue + &q_min * &modulus;
        return Some(DescentCertificate {
            word: word_string(word), residue: residue.to_string(), modulus: modulus.to_string(),
            block_length: j, odd_steps: affine.odd_count, q_min: q_min.to_string(),
            first_positive_n: first.to_string(),
        });
    }
    None
}

fn adaptive_search(word: &mut Vec<u8>, max_depth: usize, node_limit: usize,
                   nodes: &mut usize, certs: &mut Vec<DescentCertificate>,
                   unresolved: &mut Vec<UnresolvedClass>) -> Result<(), String> {
    *nodes += 1;
    if *nodes > node_limit { return Err(format!("node budget {node_limit} exceeded; increase --node-limit or reduce --max-depth")); }
    let class = ParityClass::from_word(word);
    if !verify_word(class.residue.clone(), word) {
        return Err(format!("internal parity class mismatch for {}", word_string(word)));
    }
    if !word.is_empty() {
        if let Some(cert) = find_descent(word, &class.residue, word.len()) {
            certs.push(cert);
            return Ok(());
        }
    }
    if word.len() == max_depth {
        unresolved.push(UnresolvedClass { word: word_string(word), residue: class.residue.to_string(), modulus: class.modulus.to_string() });
        return Ok(());
    }
    for bit in [0u8, 1u8] {
        word.push(bit);
        adaptive_search(word, max_depth, node_limit, nodes, certs, unresolved)?;
        word.pop();
    }
    Ok(())
}

fn parse_big(s: &str, field: &str) -> Result<BigInt, String> {
    BigInt::parse_bytes(s.as_bytes(), 10).ok_or_else(|| format!("invalid integer in {field}"))
}
fn verify_certificate(c: &ProofCertificate, base_limit: u64) -> Result<(), String> {
    if c.format != "scollatz-proof-certificate-v1" { return Err("unsupported certificate format".into()); }
    if c.max_depth == 0 || c.max_depth > 24 { return Err("invalid max_depth".into()); }
    let mut leaves: Vec<(Vec<u8>, BigInt, BigInt)> = Vec::new();
    for item in &c.descent_classes {
        let word = parse_word(&item.word)?;
        if word.is_empty() || word.len() > c.max_depth { return Err("invalid certified word depth".into()); }
        let class = ParityClass::from_word(&word);
        let residue = parse_big(&item.residue, "residue")?;
        let modulus = parse_big(&item.modulus, "modulus")?;
        if residue != class.residue || modulus != class.modulus { return Err(format!("residue mismatch for {}", item.word)); }
        if !verify_word(residue.clone(), &word) { return Err("parity word verification failed".into()); }
        if item.block_length == 0 || item.block_length > word.len() { return Err("invalid block length".into()); }
        let mut affine = Affine::identity();
        for &bit in &word[..item.block_length] { affine.append(bit); }
        if affine.odd_count != item.odd_steps { return Err("odd-step count mismatch".into()); }
        let slope = &affine.a * (BigInt::one() << (word.len() - item.block_length));
        let gap = &modulus - slope;
        if gap <= BigInt::zero() { return Err("certificate does not have a contracting block slope".into()); }
        let delta = affine.apply(&residue) - &residue;
        let q = parse_big(&item.q_min, "q_min")?;
        let first = parse_big(&item.first_positive_n, "first_positive_n")?;
        if q < BigInt::zero() { return Err("negative q_min".into()); }
        let min_q = if delta < BigInt::zero() { BigInt::zero() } else { delta / &gap + BigInt::one() };
        let min_q = if residue.is_zero() && min_q.is_zero() { BigInt::one() } else { min_q };
        if q != min_q || first != &residue + &q * &modulus { return Err("descent threshold mismatch".into()); }
        let n = &residue + &q * &modulus;
        let after = run_steps(&n, item.block_length);
        if after >= n { return Err("reported threshold does not strictly descend".into()); }
        leaves.push((word, residue, modulus));
    }
    for item in &c.unresolved_classes {
        let word = parse_word(&item.word)?;
        if word.len() != c.max_depth { return Err("unresolved classes must be at max_depth".into()); }
        let class = ParityClass::from_word(&word);
        let residue = parse_big(&item.residue, "unresolved residue")?;
        let modulus = parse_big(&item.modulus, "unresolved modulus")?;
        if residue != class.residue || modulus != class.modulus || !verify_word(residue.clone(), &word) {
            return Err("unresolved class residue/parity mismatch".into());
        }
        leaves.push((word, residue, modulus));
    }
    // Prefix-free leaves with Kraft sum 1 form a complete partition of all
    // integers by their parity cylinders through max_depth.
    let mut weight = BigInt::zero();
    for (i, (w, _, _)) in leaves.iter().enumerate() {
        weight += BigInt::one() << (c.max_depth - w.len());
        for (j, (other, _, _)) in leaves.iter().enumerate() {
            if i != j && other.len() >= w.len() && other[..w.len()] == w[..] {
                return Err("leaf set is not prefix-free".into());
            }
        }
    }
    if weight != (BigInt::one() << c.max_depth) { return Err("certificate leaves do not cover all parity classes".into()); }

    let all_certified = c.unresolved_classes.is_empty();
    let max_first = c.descent_classes.iter().map(|x| parse_big(&x.first_positive_n, "first_positive_n"))
        .collect::<Result<Vec<_>, _>>()?.into_iter().max().unwrap_or_else(BigInt::one);
    if c.complete_induction_certificate != (all_certified && c.base_verified_up_to.as_ref().is_some_and(|v| parse_big(v, "base_verified_up_to").ok() == Some(&max_first - 1))) {
        return Err("complete-induction flag/base bound is inconsistent".into());
    }
    if let Some(bound_text) = &c.base_verified_up_to {
        let bound = parse_big(bound_text, "base_verified_up_to")?;
        if bound > BigInt::from(base_limit) { return Err(format!("base verification bound exceeds verifier limit {base_limit}")); }
        let end = bound.to_u64_digits().1.first().copied().unwrap_or(0);
        for n in 1..=end {
            let mut x = BigInt::from(n);
            let mut seen = 0usize;
            while x != BigInt::one() && seen < 1_000_000 {
                x = step(&x); seen += 1;
            }
            if x != BigInt::one() { return Err(format!("base case {n} did not reach 1 within verification cap")); }
        }
    }
    Ok(())
}

fn parse_args() -> Result<(usize, usize, u64, Option<String>, Option<String>), String> {
    let mut depth=12usize; let mut node_limit=1_000_000usize; let mut base_limit=2_000_000u64;
    let mut certificate=None; let mut verify=None;
    let mut args=env::args().skip(1);
    while let Some(a)=args.next() {
        match a.as_str() {
            "--help" | "-h" => { println!("scollatz-research [--max-depth N] [--node-limit N] [--base-limit N] [--certificate FILE] | --verify FILE"); process::exit(0); }
            "--depth" | "--max-depth" => depth=args.next().ok_or("--max-depth needs N")?.parse().map_err(|_|"invalid depth")?,
            "--node-limit" => node_limit=args.next().ok_or("--node-limit needs N")?.parse().map_err(|_|"invalid node limit")?,
            "--base-limit" => base_limit=args.next().ok_or("--base-limit needs N")?.parse().map_err(|_|"invalid base limit")?,
            "--certificate" => certificate=Some(args.next().ok_or("--certificate needs FILE")?),
            "--verify" => verify=Some(args.next().ok_or("--verify needs FILE")?),
            _ => return Err(format!("unknown option: {a}")),
        }
    }
    if depth==0 || depth>24 { return Err("--max-depth must be in 1..=24".into()); }
    if node_limit==0 { return Err("--node-limit must be positive".into()); }
    if certificate.is_some() && verify.is_some() { return Err("--certificate and --verify cannot be combined".into()); }
    Ok((depth,node_limit,base_limit,certificate,verify))
}

fn main() {
    let (max_depth,node_limit,base_limit,certificate_path,verify_path)=match parse_args() {
        Ok(v)=>v, Err(e)=>{eprintln!("error: {e}");process::exit(2);}
    };
    if let Some(path)=verify_path {
        let result=(|| -> Result<(),String> {
            let data=fs::read_to_string(&path).map_err(|e|e.to_string())?;
            let cert:ProofCertificate=serde_json::from_str(&data).map_err(|e|e.to_string())?;
            verify_certificate(&cert,base_limit)?;
            println!("certificate_verification=success");
            println!("complete_induction_certificate={}",cert.complete_induction_certificate);
            println!("certified_classes={}",cert.descent_classes.len());
            println!("unresolved_classes={}",cert.unresolved_classes.len());
            Ok(())
        })();
        if let Err(e)=result { eprintln!("certificate_verification=failed: {e}");process::exit(4); }
        return;
    }

    let mut nodes=0usize; let mut certs=Vec::new(); let mut unresolved=Vec::new();
    if let Err(e)=adaptive_search(&mut Vec::new(),max_depth,node_limit,&mut nodes,&mut certs,&mut unresolved) {
        eprintln!("search_incomplete: {e}"); process::exit(5);
    }
    let all_certified=unresolved.is_empty();
    let max_first=certs.iter().map(|x|parse_big(&x.first_positive_n,"threshold").unwrap()).max().unwrap_or_else(BigInt::one);
    let base_bound=&max_first-BigInt::one();
    let base_u64=base_bound.to_u64_digits().1.first().copied().unwrap_or(0);
    let base_verified=if all_certified && base_bound>=BigInt::zero() && base_bound<=BigInt::from(base_limit) {
        let mut good=true;
        for n in 1..=base_u64 {
            let mut x=BigInt::from(n); let mut count=0usize;
            while x!=BigInt::one() && count<1_000_000 { x=step(&x);count+=1; }
            if x!=BigInt::one() { good=false;break; }
        }
        if good { Some(base_bound.to_string()) } else { None }
    } else { None };
    let complete=all_certified && base_verified.is_some();
    let proof=ProofCertificate {
        format:"scollatz-proof-certificate-v1".into(),max_depth,visited_nodes:nodes,
        descent_classes:certs,unresolved_classes:unresolved,
        base_verified_up_to:base_verified,complete_induction_certificate:complete,
    };
    println!("system=adaptive parity-class refinement + multi-block descent + induction-base verification");
    println!("max_depth={max_depth} visited_nodes={nodes}");
    println!("descent_certified_classes={}",proof.descent_classes.len());
    println!("unresolved_max_depth_classes={}",proof.unresolved_classes.len());
    println!("complete_induction_certificate={complete}");
    println!("base_limit={base_limit}");
    if let Some(path)=certificate_path {
        let serialized = match serde_json::to_string_pretty(&proof) {
            Ok(s) => s,
            Err(e) => { eprintln!("certificate_serialization_failed: {e}"); process::exit(6); }
        };
        if let Err(e) = fs::write(&path, serialized) {
            eprintln!("certificate_write_failed: {e}"); process::exit(6);
        }
        println!("certificate_file={path}");
    }
    println!("proof_boundary=only a verified complete induction certificate would establish the claim; unresolved classes are never treated as divergent");
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test] fn parity_lift_matches_word() {
        for d in 1..=9 { for mask in 0..(1u64<<d) {
            let word=(0..d).map(|i|((mask>>i)&1) as u8).collect::<Vec<_>>();
            let c=ParityClass::from_word(&word);
            assert!(verify_word(c.residue.clone(),&word));
        }}
    }
    #[test] fn multiblock_certificate_is_exact_on_class() {
        let word=parse_word("1010").unwrap(); let c=ParityClass::from_word(&word);
        if let Some(cert)=find_descent(&word,&c.residue,word.len()) {
            let q=parse_big(&cert.q_min,"q").unwrap();
            let n=&c.residue+&q*&c.modulus;
            assert!(run_steps(&n,cert.block_length)<n);
            for i in 0..10 {
                let nn=&c.residue+(&q+BigInt::from(i))*&c.modulus;
                assert!(run_steps(&nn,cert.block_length)<nn);
            }
        }
    }
    #[test] fn adaptive_partition_is_prefix_free_and_complete() {
        let mut nodes=0;let mut certs=Vec::new();let mut unresolved=Vec::new();
        adaptive_search(&mut Vec::new(),7,10000,&mut nodes,&mut certs,&mut unresolved).unwrap();
        let proof=ProofCertificate{format:"scollatz-proof-certificate-v1".into(),max_depth:7,visited_nodes:nodes,
            descent_classes:certs,unresolved_classes:unresolved,base_verified_up_to:None,complete_induction_certificate:false};
        verify_certificate(&proof,0).unwrap();
    }
    #[test] fn one_two_cycle_is_not_misreported_as_descent_for_10_prefix() {
        let w=parse_word("10").unwrap();let c=ParityClass::from_word(&w);
        let cert=find_descent(&w,&c.residue,w.len()).unwrap();
        assert_eq!(cert.first_positive_n,"5");
    }
}
