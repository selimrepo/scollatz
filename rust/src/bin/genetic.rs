#[path="../rng.rs"] mod rng;\n#[cfg(feature="opencl")] #[path="../opencl.rs"] mod opencl;
use rng::Mt19937_64;
use rayon::prelude::*;
use serde::{Deserialize,Serialize};
use std::{env,fs,process,time::Instant};

type U=u64;
#[derive(Clone,Copy,Default,Serialize,Deserialize)]
struct Fit { steps:U,peak:U,done:bool }
#[derive(Clone,Copy,Default,Serialize,Deserialize)]
struct Person { gene:U,fit:Fit }
struct Island { people:Vec<Person>,rng:Mt19937_64 }
#[derive(Serialize)]
struct Record { generation:usize,island:usize,candidate:U,stopping_time:U,peak:U,converged:bool,elapsed_seconds:f64 }
#[derive(Deserialize)]
struct CppCheckpoint { version:u32,generation:usize,seed:U,rng:Vec<String>,populations:Vec<Vec<U>> }
impl Opt { fn clone_for_threads(&self,threads:usize)->Self { Self { population:self.population,generations:self.generations,elite:self.elite,tournament:self.tournament,islands:self.islands,migrate_every:self.migrate_every,migrate_count:self.migrate_count,threads,mutation:self.mutation,limit:self.limit,lo:self.lo,hi:self.hi,seed:self.seed,backend:self.backend.clone(),checkpoint:self.checkpoint.clone(),resume:self.resume.clone(),csv:self.csv.clone(),json:self.json.clone(),adaptive:self.adaptive } } }\nstruct Opt { population:usize,generations:usize,elite:usize,tournament:usize,islands:usize,migrate_every:usize,migrate_count:usize,threads:usize,mutation:f64,limit:U,lo:U,hi:U,seed:U,backend:String,checkpoint:String,resume:String,csv:String,json:String,adaptive:bool }
fn options()->Result<Opt,String>{
 let mut o=Opt{population:512,generations:200,elite:8,tournament:5,islands:4,migrate_every:10,migrate_count:1,threads:std::thread::available_parallelism().map_or(1,usize::from),mutation:0.04,limit:1_000_000,lo:1,hi:1_000_000_000,seed:0,backend:"auto".into(),checkpoint:"scollatz.chk".into(),resume:String::new(),csv:"scollatz.csv".into(),json:"scollatz.json".into(),adaptive:true};
 let mut a=env::args().skip(1);while let Some(k)=a.next(){if k=="--help"||k=="-h"{println!("scollatz-ga --backend auto|cpu|opencl|hybrid --population N --generations N --islands N --threads N --min N --max N --max-steps N --seed N --migration-interval N --migration-count N --no-adaptive-migration --checkpoint FILE --resume FILE --csv FILE --json FILE");process::exit(0);}if k=="--no-adaptive-migration"{o.adaptive=false;continue;}let v=a.next().ok_or_else(||format!("missing value for {k}"))?;match k.as_str(){"--population"=>o.population=v.parse().map_err(|_|"population")?,"--generations"=>o.generations=v.parse().map_err(|_|"generations")?,"--elite"=>o.elite=v.parse().map_err(|_|"elite")?,"--tournament"=>o.tournament=v.parse().map_err(|_|"tournament")?,"--islands"=>o.islands=v.parse().map_err(|_|"islands")?,"--migration-interval"=>o.migrate_every=v.parse().map_err(|_|"migration interval")?,"--migration-count"=>o.migrate_count=v.parse().map_err(|_|"migration count")?,"--threads"=>o.threads=v.parse().map_err(|_|"threads")?,"--mutation"=>o.mutation=v.parse().map_err(|_|"mutation")?,"--max-steps"=>o.limit=v.parse().map_err(|_|"max steps")?,"--min"=>o.lo=v.parse().map_err(|_|"min")?,"--max"=>o.hi=v.parse().map_err(|_|"max")?,"--seed"=>o.seed=v.parse().map_err(|_|"seed")?,"--backend"=>o.backend=v,"--checkpoint"=>o.checkpoint=v,"--resume"=>o.resume=v,"--csv"=>o.csv=v,"--json"=>o.json=v,_=>return Err(format!("unknown option {k}"))}}
 if o.population<8||o.generations==0||o.islands==0||o.islands>o.population||o.elite==0||o.elite>=o.population/o.islands||o.tournament==0||o.threads==0||o.limit==0||o.lo==0||o.hi<o.lo||o.migrate_every==0||o.migrate_count==0||!(0.0..=1.0).contains(&o.mutation)||!["auto","cpu","opencl","hybrid"].contains(&o.backend.as_str()){return Err("invalid GA configuration".into());}Ok(o)
}
fn fitness(mut n:U,limit:U)->Fit{let mut steps=0;let mut peak=n;while n!=1&&steps<limit{if n&1==0{n>>=1;}else{if n>(U::MAX-1)/3{return Fit{steps,peak,done:false};}n=n*3+1;}steps+=1;peak=peak.max(n);}Fit{steps,peak,done:n==1}}
fn better(a:&Person,b:&Person)->bool{if a.fit.done!=b.fit.done{return a.fit.done;}if a.fit.steps!=b.fit.steps{return a.fit.steps>b.fit.steps;}if a.fit.peak!=b.fit.peak{return a.fit.peak>b.fit.peak;}a.gene<b.gene}
fn mutate(mut x:U,o:&Opt,r:&mut Mt19937_64)->U{for bit in 0..64{if r.bernoulli(o.mutation){x^=1u64<<bit;}}if r.bernoulli(o.mutation){let d=r.uniform(0,((o.hi-o.lo)/32).max(1));if r.bernoulli(o.mutation){x=if x>d{x-d}else{o.lo};}else if U::MAX-x>=d{x+=d;}}x.clamp(o.lo,o.hi)}
fn tournament(p:&[Person],n:usize,r:&mut Mt19937_64)->Person{let mut b=p[r.uniform(0,(p.len()-1) as U) as usize];let mut left=n.saturating_sub(1);while left>0{let j=r.uniform(0,(p.len()-1) as U) as usize;let c=p[j];if better(&c,&b){b=c;}left-=1;}b}
fn evaluate_cpu(p:&mut[Person],o:&Opt){let pool=rayon::ThreadPoolBuilder::new().num_threads(o.threads).build().expect("GA worker pool");pool.install(||p.par_iter_mut().for_each(|x|x.fit=fitness(x.gene,o.limit)));}
#[cfg(feature="opencl")]
fn evaluate_backend(p:&mut[Person],o:&Opt,gpu:Option<&mut opencl::Gpu>)->Result<(),String>{
 match o.backend.as_str(){
  "cpu"=>{evaluate_cpu(p,o);Ok(())},
  "opencl"|"auto"=>{let g=gpu.ok_or_else(||"OpenCL executor unavailable".to_string())?;let genes=p.iter().map(|x|x.gene).collect::<Vec<_>>();let out=g.candidates(&genes,o.limit)?;for (person,(steps,peak,status)) in p.iter_mut().zip(out){person.fit=Fit{steps,peak,done:status==0};}Ok(())},
  "hybrid"=>{let g=gpu.ok_or_else(||"OpenCL executor unavailable".to_string())?;let split=(p.len()+1)/2;let (gpu_part,cpu_part)=p.split_at_mut(split);std::thread::scope(|scope|{let worker=scope.spawn(||evaluate_cpu(cpu_part,o));let genes=gpu_part.iter().map(|x|x.gene).collect::<Vec<_>>();let gpu_out=g.candidates(&genes,o.limit);worker.join().map_err(|_|"CPU worker panicked".to_string())?;let out=gpu_out?;for (person,(steps,peak,status)) in gpu_part.iter_mut().zip(out){person.fit=Fit{steps,peak,done:status==0};}Ok(())})},
  _=>Err("unsupported backend".into())
 }
}
#[cfg(not(feature="opencl"))]
fn evaluate_backend(p:&mut[Person],o:&Opt)->Result<(),String>{if o.backend=="opencl"{return Err("Rust OpenCL support not compiled; rebuild with --features opencl".into());}evaluate_cpu(p,o);Ok(())}
fn write_checkpoint(path:&str,generation:usize,seed:U,islands:&[Island])->Result<(),String>{
 let rng=islands.iter().map(|i|i.rng.to_cpp_state()).collect::<Vec<_>>();
 let populations=islands.iter().map(|i|i.people.iter().map(|p|p.gene).collect()).collect();
 let value=serde_json::json!({"version":1,"generation":generation,"seed":seed,"rng":rng,"populations":populations});
 fs::write(path,serde_json::to_vec_pretty(&value).map_err(|e|e.to_string())?).map_err(|e|format!("checkpoint write failed: {e}"))
}
fn read_checkpoint(path:&str,islands:&mut[Island],o:&Opt)->Result<(usize,U),String>{
 let raw=fs::read_to_string(path).map_err(|e|format!("cannot read checkpoint: {e}"))?;
 let c:CppCheckpoint=serde_json::from_str(&raw).map_err(|e|format!("invalid C++ checkpoint: {e}"))?;
 if c.version!=1||c.rng.len()!=islands.len()||c.populations.len()!=islands.len(){return Err("unsupported checkpoint schema or island count".into());}
 for i in 0..islands.len(){let expected=o.population/o.islands+usize::from(i<o.population%o.islands);if c.populations[i].len()!=expected{return Err("checkpoint population dimensions do not match --population/--islands".into());}islands[i].rng=Mt19937_64::from_cpp_state(&c.rng[i])?;islands[i].people=c.populations[i].iter().map(|&gene|{if gene<o.lo||gene>o.hi{return Err("checkpoint gene outside requested range".to_string());}Ok(Person{gene,fit:Fit::default()})}).collect::<Result<Vec<_>,String>>()?;}
 Ok((c.generation,c.seed))
}
fn outputs(o:&Opt,records:&[Record],seed:U)->Result<(),String>{
 if !o.csv.is_empty(){let mut s="generation,island,candidate,stopping_time,peak,converged,elapsed_seconds\n".to_string();for r in records{s.push_str(&format!("{},{},{},{},{},{},{}\n",r.generation,r.island,r.candidate,r.stopping_time,r.peak,r.converged,r.elapsed_seconds));}fs::write(&o.csv,s).map_err(|e|e.to_string())?;}
 if !o.json.is_empty(){#[derive(Serialize)]struct Out<'a>{seed:U,results:&'a[Record]}fs::write(&o.json,serde_json::to_vec_pretty(&Out{seed,results:records}).map_err(|e|e.to_string())?).map_err(|e|e.to_string())?;}Ok(())
}
fn main(){
 let o=match options(){Ok(x)=>x,Err(e)=>{eprintln!("Invalid arguments: {e}");process::exit(2)}};
 let mut seed=if o.seed==0{std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap().as_nanos() as U}else{o.seed};
 let mut islands:Vec<Island>=(0..o.islands).map(|i|Island{people:Vec::new(),rng:Mt19937_64::seeded(seed.wrapping_add(0x9e3779b97f4a7c15u64.wrapping_mul(i as U+1)))}).collect();
 let mut first=0usize;
 if !o.resume.is_empty(){match read_checkpoint(&o.resume,&mut islands,&o){Ok((g,s))=>{first=g;seed=s;if o.seed!=0&&o.seed!=seed{eprintln!("Warning: --seed ignored; checkpoint seed is {seed}");}},Err(e)=>{eprintln!("{e}");process::exit(2)}}}
 else{for (i,island) in islands.iter_mut().enumerate(){let n=o.population/o.islands+usize::from(i<o.population%o.islands);island.people=(0..n).map(|_|Person{gene:island.rng.uniform(o.lo,o.hi),fit:Fit::default()}).collect();}}
 let mut active_backend=o.backend.clone();
 #[cfg(feature="opencl")]
 let mut gpus:Vec<Option<opencl::Gpu>>=if active_backend=="cpu"{std::iter::repeat_with(||None).take(islands.len()).collect()}else{
   let mut all=Vec::with_capacity(islands.len());let mut failure=None;
   for island in &islands{match opencl::Gpu::new(island.people.len()){Ok(g)=>{if all.is_empty(){println!("OpenCL device: {}",g.name());}all.push(Some(g));},Err(e)=>{failure=Some(e);break;}}}
   if let Some(e)=failure{if active_backend=="opencl"{eprintln!("{e}");process::exit(3);}eprintln!("OpenCL unavailable ({e}); falling back to CPU.");active_backend="cpu".into();std::iter::repeat_with(||None).take(islands.len()).collect()}else{all}
 };
 #[cfg(not(feature="opencl"))]
 let mut active_backend={if active_backend=="opencl"{eprintln!("OpenCL support not compiled; rebuild with --features opencl.");process::exit(3);}if active_backend=="auto"||active_backend=="hybrid"{"cpu".to_string()}else{active_backend}};
 let mut records:Vec<Record>=Vec::new();let mut best=Person::default();let mut have=false;
 let threads_per_island=(o.threads/islands.len()).max(1);
 for gen in first..o.generations{
  let t=Instant::now();
  for (idx,island) in islands.iter_mut().enumerate(){
   let mut local=o.clone_for_threads(threads_per_island);\n   local.backend=active_backend.clone();
   #[cfg(feature="opencl")]
   let result=evaluate_backend(&mut island.people,&local,gpus[idx].as_mut());
   #[cfg(not(feature="opencl"))]
   let result=evaluate_backend(&mut island.people,&local);
   if let Err(e)=result{if active_backend=="opencl"{eprintln!("{e}");process::exit(3);}eprintln!("OpenCL execution failed ({e}); recomputing island on CPU.");evaluate_cpu(&mut island.people,&local);active_backend="cpu".into();}
   island.people.sort_by(|a,b|if better(a,b){std::cmp::Ordering::Less}else if better(b,a){std::cmp::Ordering::Greater}else{std::cmp::Ordering::Equal});
   if !have||better(&island.people[0],&best){best=island.people[0];have=true;}
  }
  for (i,p) in islands.iter().enumerate(){let x=p.people[0];records.push(Record{generation:gen,island:i,candidate:x.gene,stopping_time:x.fit.steps,peak:x.fit.peak,converged:x.fit.done,elapsed_seconds:0.0});}
  let mut interval=o.migrate_every;let mut migrants=o.migrate_count;let mut diversity=0.0;
  if o.adaptive{for p in &islands{let mut g:Vec<U>=p.people.iter().map(|x|x.gene).collect();g.sort_unstable();g.dedup();diversity+=g.len() as f64/p.people.len() as f64;}diversity/=islands.len() as f64;if diversity<0.25{interval=(o.migrate_every/2).max(1);migrants=(o.migrate_count*2).min((o.population/o.islands/8).max(1));}else if diversity>0.70{interval=o.migrate_every.saturating_mul(2);migrants=(o.migrate_count/2).max(1);}}
  if islands.len()>1&&(gen+1)%interval==0{let transfers:Vec<Vec<Person>>=(0..islands.len()).map(|i|islands[i].people.iter().take(migrants.min(islands[i].people.len())).copied().collect()).collect();for i in 0..islands.len(){let dst=(i+1)%islands.len();let n=transfers[i].len().min(islands[dst].people.len());let len=islands[dst].people.len();for j in 0..n{islands[dst].people[len-1-j]=transfers[i][j];}}println!("migration generation={gen} diversity={diversity} interval={interval} migrants_per_island={migrants}");}
  if gen+1<o.generations{for island in &mut islands{let old=&island.people;let mut next=Vec::with_capacity(old.len());for p in old.iter().take(o.elite.min(old.len()-1)){next.push(*p);}while next.len()<old.len(){let a=tournament(old,o.tournament,&mut island.rng);let b=tournament(old,o.tournament,&mut island.rng);let mask=island.rng.next_u64();let child=(a.gene&mask)|(b.gene&!mask);next.push(Person{gene:mutate(child,&o,&mut island.rng),fit:Fit::default()});}island.people=next;}}
  let elapsed=t.elapsed().as_secs_f64();for r in records.iter_mut().rev().take(islands.len()){r.elapsed_seconds=elapsed;}
  if let Err(e)=write_checkpoint(&o.checkpoint,gen+1,seed,&islands){eprintln!("{e}");process::exit(3);}
  println!("generation={gen} best={} stopping_time={} backend={} islands={}",best.gene,best.fit.steps,active_backend,islands.len());
 }
 if let Err(e)=outputs(&o,&records,seed){eprintln!("output write failed: {e}");process::exit(3);}
 println!("best_candidate={} stopping_time={} peak={} seed={seed}",best.gene,best.fit.steps,best.fit.peak);
 if !best.fit.done{process::exit(1);}
}
#[cfg(test)]mod tests{use super::*;#[test]fn collatz_reference(){assert_eq!(fitness(27,1000).steps,111);assert!(fitness(27,1000).done);}#[test]fn overflow_guard(){assert!(!fitness(U::MAX,100).done);}}
