use rayon::prelude::*;
#[cfg(feature="opencl")] mod opencl;
use serde::Serialize;
use std::{env, process, time::Instant};

#[derive(Clone)]
struct Opt { start:u64, end:u64, max_steps:u64, chunk:u64, threads:usize, backend:String }
#[derive(Clone, Copy, Default, Serialize)]
struct ResultSet { tested:u64, completed:u64, overflow:u64, capped:u64, max_steps:u64, max_steps_start:u64, max_peak:u64, max_peak_start:u64 }
#[derive(Clone, Copy)]
struct Sample { steps:u64, peak:u64, status:u8 }

fn parse() -> Result<Opt,String> {
    let mut o=Opt{start:1,end:1_000_000,max_steps:10_000_000,chunk:4096,threads:std::thread::available_parallelism().map_or(1,usize::from),backend:"auto".into()};
    let mut a=env::args().skip(1);
    while let Some(k)=a.next() {
        if k=="--help" || k=="-h" { println!("scollatz --start N --end N [--backend auto|cpu|opencl] [--threads N] [--chunk N] [--max-steps N]"); process::exit(0); }
        let v=a.next().ok_or_else(||format!("missing value for {k}"))?;
        match k.as_str() {
            "--start"=>o.start=v.parse().map_err(|_|"invalid --start")?,
            "--end"=>o.end=v.parse().map_err(|_|"invalid --end")?,
            "--max-steps"=>o.max_steps=v.parse().map_err(|_|"invalid --max-steps")?,
            "--chunk"=>o.chunk=v.parse().map_err(|_|"invalid --chunk")?,
            "--threads"=>o.threads=v.parse().map_err(|_|"invalid --threads")?,
            "--backend"=>o.backend=v,
            _=>return Err(format!("unknown option {k}")),
        }
    }
    if o.start==0 || o.end<o.start || o.max_steps==0 || o.chunk==0 || o.threads==0 || o.threads>4096 || !["auto","cpu","opencl"].contains(&o.backend.as_str()) { return Err("invalid range, limits, thread count, or backend".into()); }
    Ok(o)
}
fn evaluate(mut n:u64, limit:u64)->Sample {
    let mut steps=0; let mut peak=n;
    while n!=1 && steps<limit {
        if n&1==0 { n >>= 1; }
        else { if n>(u64::MAX-1)/3 { return Sample{steps,peak,status:1}; } n=n*3+1; }
        steps+=1; peak=peak.max(n);
    }
    Sample{steps,peak,status:if n==1 {0}else{2}}
}
fn merge(r:&mut ResultSet,start:u64,s:Sample) {
    r.tested+=1;
    match s.status {
        0=>{r.completed+=1;if s.steps>r.max_steps{r.max_steps=s.steps;r.max_steps_start=start;}if s.peak>r.max_peak{r.max_peak=s.peak;r.max_peak_start=start;}},
        1=>r.overflow+=1,
        _=>r.capped+=1,
    }
}
fn main() {
    let o=match parse(){Ok(x)=>x,Err(e)=>{eprintln!("Invalid arguments: {e}. Use --help.");process::exit(2)}};
    
    let start=Instant::now();
    let count=o.end-o.start+1;
    let chunks:Vec<(u64,u64)>=(0..count.div_ceil(o.chunk)).map(|i|{let b=o.start+i*o.chunk;(b,o.chunk.min(o.end-b+1))}).collect();
    let pool=rayon::ThreadPoolBuilder::new().num_threads(o.threads).build().expect("thread pool");
    let use_gpu = o.backend=="opencl" || o.backend=="auto";
    #[cfg(feature="opencl")]
    let gpu_result = if use_gpu {
        match opencl::Gpu::new(o.chunk.min(count) as usize) {
            Ok(mut gpu) => {
                println!("OpenCL device: {}",gpu.name());
                let mut total=ResultSet::default();
                let mut begin=o.start;
                while begin<=o.end {
                    let n=o.chunk.min(o.end-begin+1) as usize;
                    match gpu.range(begin,n,o.max_steps) {
                        Ok(batch)=>for (i,(steps,peak,status)) in batch.into_iter().enumerate(){merge(&mut total,begin+i as u64,Sample{steps,peak,status:status as u8});},
                        Err(e)=>{if o.backend=="opencl"{eprintln!("{e}");process::exit(3);}eprintln!("OpenCL failed ({e}); restarting scan on CPU.");total=ResultSet::default();break;}
                    }
                    if n as u64==o.end-begin+1{break;}begin+=n as u64;
                }
                if total.tested==count {Some(total)} else {None}
            },
            Err(e)=>{if o.backend=="opencl"{eprintln!("{e}");process::exit(3);}eprintln!("OpenCL unavailable ({e}); using CPU.");None}
        }
    } else {None};
    #[cfg(not(feature="opencl"))]
    let gpu_result:Option<ResultSet> = { if o.backend=="opencl" {eprintln!("OpenCL support not compiled; rebuild with --features opencl.");process::exit(3);} None };
    let partial:Vec<ResultSet>=if let Some(result)=gpu_result {vec![result]} else {pool.install(||chunks.par_iter().map(|&(b,n)|{let mut r=ResultSet::default();for x in b..b+n{merge(&mut r,x,evaluate(x,o.max_steps));}r}).collect())};
    let mut r=ResultSet::default();
    for p in partial {r.tested+=p.tested;r.completed+=p.completed;r.overflow+=p.overflow;r.capped+=p.capped;if p.max_steps>r.max_steps{r.max_steps=p.max_steps;r.max_steps_start=p.max_steps_start;}if p.max_peak>r.max_peak{r.max_peak=p.max_peak;r.max_peak_start=p.max_peak_start;}}
    let sec=start.elapsed().as_secs_f64();
    println!("Backend: {}
Tested: {} | completed: {} | overflow: {} | step-limit: {}
Longest stopping time: {} (start {})
Highest peak: {} (start {})
Elapsed: {:.3} s",if gpu_result.is_some() {"OpenCL GPU".to_string()} else {format!("CPU ({} threads)",o.threads)},r.tested,r.completed,r.overflow,r.capped,r.max_steps,r.max_steps_start,r.max_peak,r.max_peak_start,sec);
    if sec>0.0 {println!("Throughput: {:.0} starts/s",r.tested as f64/sec);}
    if r.capped>0 || r.overflow>0 {process::exit(1);}
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test] fn known_stopping_times(){assert_eq!(evaluate(1,100).steps,0);assert_eq!(evaluate(3,100).steps,7);assert_eq!(evaluate(27,1000).steps,111);}
    #[test] fn step_limit_is_reported(){assert_eq!(evaluate(27,1).status,2);}
    #[test] fn range_edges(){assert_eq!(evaluate(2,10).status,0);assert_eq!(evaluate(0,10).status,2);}
}
