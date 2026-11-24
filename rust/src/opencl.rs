use ocl::{flags::MemFlags, Buffer, Device, DeviceType, Platform, ProQue};

const KERNEL: &str = include_str!("../../kernels/collatz.cl");

pub struct Gpu {
    pro_que: ProQue,
    capacity: usize,
    candidates: Buffer<u64>,
    steps: Buffer<u64>,
    peaks: Buffer<u64>,
    status: Buffer<u32>,
}
impl Gpu {
    pub fn new(capacity: usize) -> Result<Self,String> {
        let platforms=Platform::list();
        if platforms.is_empty(){return Err("No OpenCL platform found".into());}
        let mut selected=None;
        for p in &platforms {
            if let Ok(ds)=Device::list(*p,Some(DeviceType::GPU)) {
                if let Some(d)=ds.first(){selected=Some((*p,*d));break;}
            }
        }
        if selected.is_none() {
            for p in &platforms {
                if let Ok(ds)=Device::list_all(*p) {
                    if let Some(d)=ds.first(){selected=Some((*p,*d));break;}
                }
            }
        }
        let (platform,device)=selected.ok_or_else(||"No OpenCL device found".to_string())?;
        let pq=ProQue::builder().platform(platform).device(device).src(KERNEL).dims(capacity.max(1)).build().map_err(|e|format!("OpenCL program initialization failed: {e}"))?;
        let queue=pq.queue().clone();
        let candidates=Buffer::<u64>::builder().queue(queue.clone()).flags(MemFlags::READ_ONLY).len(capacity.max(1)).build().map_err(|e|e.to_string())?;
        let steps=Buffer::<u64>::builder().queue(queue.clone()).flags(MemFlags::WRITE_ONLY).len(capacity.max(1)).build().map_err(|e|e.to_string())?;
        let peaks=Buffer::<u64>::builder().queue(queue.clone()).flags(MemFlags::WRITE_ONLY).len(capacity.max(1)).build().map_err(|e|e.to_string())?;
        let status=Buffer::<u32>::builder().queue(queue).flags(MemFlags::WRITE_ONLY).len(capacity.max(1)).build().map_err(|e|e.to_string())?;
        Ok(Self{pro_que:pq,capacity:capacity.max(1),candidates,steps,peaks,status})
    }
    pub fn name(&self)->String { self.pro_que.device().name().unwrap_or_else(|_|"OpenCL device".into()) }
    pub fn candidates(&mut self,values:&[u64],limit:u64)->Result<Vec<(u64,u64,u32)>,String>{
        let n=values.len();if n>self.capacity{return Err("OpenCL batch exceeds allocated capacity".into());}if n==0{return Ok(Vec::new());}
        self.candidates.write(values).enq().map_err(|e|format!("OpenCL candidate upload: {e}"))?;
        let kernel=self.pro_que.kernel_builder("collatz_candidates")
            .arg(&self.candidates).arg(n as u64).arg(&self.steps).arg(&self.peaks).arg(&self.status).arg(limit)
            .build().map_err(|e|format!("OpenCL kernel build: {e}"))?;
        unsafe{kernel.cmd().global_work_size(n).enq().map_err(|e|format!("OpenCL dispatch: {e}"))?;}
        let mut s=vec![0u64;n];let mut p=vec![0u64;n];let mut st=vec![0u32;n];
        self.steps.read(&mut s).enq().map_err(|e|e.to_string())?;
        self.peaks.read(&mut p).enq().map_err(|e|e.to_string())?;
        self.status.read(&mut st).enq().map_err(|e|e.to_string())?;
        Ok((0..n).map(|i|(s[i],p[i],st[i])).collect())
    }
    pub fn range(&mut self,first:u64,count:usize,limit:u64)->Result<Vec<(u64,u64,u32)>,String>{
        let n=count;if n>self.capacity{return Err("OpenCL range batch exceeds allocated capacity".into());}if n==0{return Ok(Vec::new());}
        let kernel=self.pro_que.kernel_builder("collatz_batch")
            .arg(first).arg(n as u64).arg(&self.steps).arg(&self.peaks).arg(&self.status).arg(limit)
            .build().map_err(|e|format!("OpenCL kernel build: {e}"))?;
        unsafe{kernel.cmd().global_work_size(n).enq().map_err(|e|format!("OpenCL dispatch: {e}"))?;}
        let mut s=vec![0u64;n];let mut p=vec![0u64;n];let mut st=vec![0u32;n];
        self.steps.read(&mut s).enq().map_err(|e|e.to_string())?;
        self.peaks.read(&mut p).enq().map_err(|e|e.to_string())?;
        self.status.read(&mut st).enq().map_err(|e|e.to_string())?;
        Ok((0..n).map(|i|(s[i],p[i],st[i])).collect())
    }
}
