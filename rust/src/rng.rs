// Exact, portable implementation of C++ std::mt19937_64 plus libstdc++
// uniform_int_distribution's 64-bit Lemire downscaling and bernoulli rule.
// This makes seeded GA runs reproducible across the C++ and Rust binaries.
#[derive(Clone)]
pub struct Mt19937_64 { state: [u64; 312], index: usize }

impl Mt19937_64 {
    pub fn seeded(seed: u64) -> Self {
        let mut state = [0u64; 312];
        state[0] = seed;
        for i in 1..312 {
            let x = state[i-1] ^ (state[i-1] >> 62);
            state[i] = 6364136223846793005u64.wrapping_mul(x).wrapping_add(i as u64);
        }
        Self { state, index: 312 }
    }
    pub fn from_cpp_state(s: &str) -> Result<Self, String> {
        let values: Vec<u64> = s.split_whitespace().map(|x| x.parse::<u64>().map_err(|_| "invalid MT19937-64 state".to_string())).collect::<Result<_,_>>()?;
        if values.len() != 313 || values[312] > 312 { return Err("invalid MT19937-64 state length/index".into()); }
        let mut state=[0u64;312]; state.copy_from_slice(&values[..312]);
        Ok(Self { state, index: values[312] as usize })
    }
    pub fn to_cpp_state(&self) -> String {
        let mut out=String::with_capacity(6500);
        for (i,v) in self.state.iter().enumerate() { if i>0 {out.push(' ');} out.push_str(&v.to_string()); }
        out.push(' '); out.push_str(&self.index.to_string()); out
    }
    fn twist(&mut self) {
        const A:u64=0xB5026F5AA96619E9;
        for i in 0..312 {
            let x=(self.state[i]&0xFFFFFFFF80000000)|(self.state[(i+1)%312]&0x7FFFFFFF);
            let xa=if x&1!=0 {A}else{0};
            self.state[i]=self.state[(i+156)%312]^(x>>1)^xa;
        }
        self.index=0;
    }
    pub fn next_u64(&mut self)->u64 {
        if self.index>=312 {self.twist();}
        let mut x=self.state[self.index];self.index+=1;
        x^=(x>>29)&0x5555555555555555;
        x^=(x<<17)&0x71D67FFFEDA60000;
        x^=(x<<37)&0xFFF7EEE000000000;
        x^=x>>43;x
    }
    // Equivalent to uniform_int_distribution<uint64_t>(a,b) in GCC/libstdc++.
    pub fn uniform(&mut self,a:u64,b:u64)->u64 {
        let range=b.wrapping_sub(a).wrapping_add(1);
        if range==0 {return self.next_u64();}
        loop {
            let product=(self.next_u64() as u128)*(range as u128);
            let low=product as u64;
            if low<range {
                let threshold=range.wrapping_neg()%range;
                if low<threshold {continue;}
            }
            return a.wrapping_add((product>>64) as u64);
        }
    }
    // libstdc++ bernoulli_distribution uses generate_canonical<double, 53>.
    pub fn bernoulli(&mut self,p:f64)->bool {
        let x=self.next_u64();
        if p<=0.0{return false;} if p>=1.0{return true;}
        let unit=(x as f64)*(1.0/18446744073709551616.0);
        unit<p
    }
}

#[cfg(test)]
mod tests {
 use super::*;
 #[test] fn reference_mt19937_64_sequence(){
  let mut r=Mt19937_64::seeded(5489);
  assert_eq!(r.next_u64(),14514284786278117030);
  assert_eq!(r.next_u64(),4620546740167642908);
  assert_eq!(r.next_u64(),13109570281517897720);
 }
 #[test] fn cpp_state_round_trip(){let mut a=Mt19937_64::seeded(42);for _ in 0..319{a.next_u64();}let s=a.to_cpp_state();let mut b=Mt19937_64::from_cpp_state(&s).unwrap();assert_eq!(a.next_u64(),b.next_u64());}
 #[test] fn gcc_reference_uniform_sequence(){let mut r=Mt19937_64::seeded(42);let expected=[755155533,639031394,752145201,136272684,903268967,94068312,574570305,372887700];for v in expected{assert_eq!(r.uniform(1,1_000_000_000),v);}}
 #[test] fn bernoulli_consumes_one_engine_value_at_endpoints(){let mut a=Mt19937_64::seeded(42);assert!(!a.bernoulli(0.0));let mut b=Mt19937_64::seeded(42);b.next_u64();assert_eq!(a.next_u64(),b.next_u64());}

}
