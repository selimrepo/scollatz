#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <memory>
#include <random>
#include <regex>
#include <sstream>
#include <string>
#include <thread>
#include <vector>
#ifdef SCOLLATZ_HAS_OPENCL
#define CL_TARGET_OPENCL_VERSION 120
#include <CL/cl.h>
#endif
namespace {
using U=std::uint64_t;
struct Fit{U steps=0,peak=0;bool done=false;};
struct Person{U gene=1;Fit fit{};};
struct Opt{size_t population=512,generations=200,elite=8,tournament=5,islands=4,migrate_every=10,migrate_count=1;unsigned threads=std::max(1u,std::thread::hardware_concurrency());double mutation=.04;U limit=1000000,lo=1,hi=1000000000,seed=0;std::string backend="auto",checkpoint="scollatz.chk",resume,csv="scollatz.csv",json="scollatz.json";};
struct Island{std::vector<Person> p;std::mt19937_64 rng;};
struct Rec{size_t gen,island;Person best;double seconds;};
bool num(const char*s,U&v){try{size_t n=0;auto x=std::stoull(s,&n);if(s[n])return false;v=x;return true;}catch(...){return false;}}
bool sz(const char*s,size_t&v){U x;if(!num(s,x)||x>SIZE_MAX)return false;v=(size_t)x;return true;}
bool dbl(const char*s,double&v){try{size_t n=0;v=std::stod(s,&n);return !s[n]&&v>=0&&v<=1;}catch(...){return false;}}
bool parse(int ac,char**av,Opt&o){for(int i=1;i<ac;++i){std::string a=av[i];if(a=="--help"||a=="-h"){std::cout<<"scollatz-ga --backend auto|cpu|opencl|hybrid --islands N --population N --generations N --migration-interval N --migration-count N --checkpoint FILE --resume FILE --csv FILE --json FILE [--threads N --min N --max N --seed N]\\n";std::exit(0);}if(i+1>=ac)return false;const char*v=av[++i];
if(a=="--population"){if(!sz(v,o.population))return false;}else if(a=="--generations"){if(!sz(v,o.generations))return false;}else if(a=="--elite"){if(!sz(v,o.elite))return false;}else if(a=="--tournament"){if(!sz(v,o.tournament))return false;}else if(a=="--islands"){if(!sz(v,o.islands))return false;}else if(a=="--migration-interval"){if(!sz(v,o.migrate_every))return false;}else if(a=="--migration-count"){if(!sz(v,o.migrate_count))return false;}else if(a=="--threads"){U x;if(!num(v,x)||!x||x>4096)return false;o.threads=(unsigned)x;}else if(a=="--mutation"){if(!dbl(v,o.mutation))return false;}else if(a=="--max-steps"){if(!num(v,o.limit))return false;}else if(a=="--min"){if(!num(v,o.lo))return false;}else if(a=="--max"){if(!num(v,o.hi))return false;}else if(a=="--seed"){if(!num(v,o.seed))return false;}else if(a=="--backend")o.backend=v;else if(a=="--checkpoint")o.checkpoint=v;else if(a=="--resume")o.resume=v;else if(a=="--csv")o.csv=v;else if(a=="--json")o.json=v;else return false;}
return o.population>=8&&o.generations&&o.islands&&o.islands<=o.population&&o.elite&&o.elite<o.population/o.islands&&o.tournament&&o.limit&&o.lo&&o.hi>=o.lo&&o.migrate_every&&o.migrate_count&&(o.backend=="auto"||o.backend=="cpu"||o.backend=="opencl"||o.backend=="hybrid");}
Fit fitness(U n,U lim){U s=0,p=n;while(n!=1&&s<lim){if(!(n&1))n>>=1;else{if(n>(UINT64_MAX-1)/3)return{s,p,false};n=n*3+1;}++s;p=std::max(p,n);}return{s,p,n==1};}
bool better(const Person&a,const Person&b){if(a.fit.done!=b.fit.done)return a.fit.done;if(a.fit.steps!=b.fit.steps)return a.fit.steps>b.fit.steps;if(a.fit.peak!=b.fit.peak)return a.fit.peak>b.fit.peak;return a.gene<b.gene;}
U random_u(std::mt19937_64&r,U a,U b){return std::uniform_int_distribution<U>(a,b)(r);}
U mutate(U x,const Opt&o,std::mt19937_64&r){std::bernoulli_distribution f(o.mutation);for(unsigned b=0;b<64;++b)if(f(r))x^=U(1)<<b;if(f(r)){U d=random_u(r,0,std::max<U>(1,(o.hi-o.lo)/32));if(f(r))x=x>d?x-d:o.lo;else if(UINT64_MAX-x>=d)x+=d;}return std::clamp(x,o.lo,o.hi);}
const Person& tournament(const std::vector<Person>&p,std::mt19937_64&r,size_t n){size_t b=random_u(r,0,p.size()-1);while(--n){size_t j=random_u(r,0,p.size()-1);if(better(p[j],p[b]))b=j;}return p[b];}
void cpu(std::vector<Person>&p,size_t b,size_t e,const Opt&o){std::atomic<size_t>next{b};std::vector<std::thread>w;unsigned n=std::min<unsigned>(o.threads,(unsigned)(e-b));for(unsigned t=0;t<n;++t)w.emplace_back([&]{for(;;){size_t i=next++;if(i>=e)break;p[i].fit=fitness(p[i].gene,o.limit);}});for(auto&t:w)t.join();}
#ifdef SCOLLATZ_HAS_OPENCL
class GPU{cl_context c=nullptr;cl_command_queue q=nullptr;cl_program prog=nullptr;cl_kernel k=nullptr;
public:GPU(){cl_uint np=0;if(clGetPlatformIDs(0,nullptr,&np)!=CL_SUCCESS||!np)throw std::runtime_error("No OpenCL platform");std::vector<cl_platform_id>ps(np);clGetPlatformIDs(np,ps.data(),nullptr);cl_device_id d=nullptr;for(auto type:{CL_DEVICE_TYPE_GPU,CL_DEVICE_TYPE_ALL}){for(auto p:ps)if(clGetDeviceIDs(p,type,1,&d,nullptr)==CL_SUCCESS)break;if(d)break;}if(!d)throw std::runtime_error("No OpenCL device");cl_int e;c=clCreateContext(nullptr,1,&d,nullptr,nullptr,&e);if(e)throw std::runtime_error("OpenCL context failure");q=clCreateCommandQueue(c,d,0,&e);if(e)throw std::runtime_error("OpenCL queue failure");std::ifstream f(SCOLLATZ_GA_KERNEL_PATH);if(!f)throw std::runtime_error("Missing GA kernel");std::ostringstream ss;ss<<f.rdbuf();auto s=ss.str();const char*ptr=s.c_str();size_t len=s.size();prog=clCreateProgramWithSource(c,1,&ptr,&len,&e);if(!e)e=clBuildProgram(prog,1,&d,nullptr,nullptr,nullptr);if(e){size_t z=0;clGetProgramBuildInfo(prog,d,CL_PROGRAM_BUILD_LOG,0,nullptr,&z);std::string log(z,' ');if(z)clGetProgramBuildInfo(prog,d,CL_PROGRAM_BUILD_LOG,z,log.data(),nullptr);throw std::runtime_error("OpenCL build: "+log);}k=clCreateKernel(prog,"collatz_candidates",&e);if(e)throw std::runtime_error("OpenCL kernel creation failed");char name[256]{};clGetDeviceInfo(d,CL_DEVICE_NAME,sizeof(name),name,nullptr);std::cout<<"OpenCL fitness device: "<<name<<"\\n";}
~GPU(){if(k)clReleaseKernel(k);if(prog)clReleaseProgram(prog);if(q)clReleaseCommandQueue(q);if(c)clReleaseContext(c);}
void run(std::vector<Person>&p,size_t b,size_t e,U lim){if(b==e)return;size_t n=e-b;std::vector<cl_ulong>in(n),st(n),pk(n);std::vector<cl_uint>status(n);for(size_t i=0;i<n;++i)in[i]=p[b+i].gene;cl_int err;cl_mem a=clCreateBuffer(c,CL_MEM_READ_ONLY|CL_MEM_COPY_HOST_PTR,n*sizeof(cl_ulong),in.data(),&err);cl_mem x=clCreateBuffer(c,CL_MEM_WRITE_ONLY,n*sizeof(cl_ulong),nullptr,&err),y=clCreateBuffer(c,CL_MEM_WRITE_ONLY,n*sizeof(cl_ulong),nullptr,&err),z=clCreateBuffer(c,CL_MEM_WRITE_ONLY,n*sizeof(cl_uint),nullptr,&err);if(err)throw std::runtime_error("OpenCL buffers failed");cl_ulong count=n,limit=lim;clSetKernelArg(k,0,sizeof(a),&a);clSetKernelArg(k,1,sizeof(count),&count);clSetKernelArg(k,2,sizeof(x),&x);clSetKernelArg(k,3,sizeof(y),&y);clSetKernelArg(k,4,sizeof(z),&z);clSetKernelArg(k,5,sizeof(limit),&limit);size_t global=n;err=clEnqueueNDRangeKernel(q,k,1,nullptr,&global,nullptr,0,nullptr,nullptr);if(!err)err=clEnqueueReadBuffer(q,x,CL_TRUE,0,n*sizeof(cl_ulong),st.data(),0,nullptr,nullptr);if(!err)err=clEnqueueReadBuffer(q,y,CL_TRUE,0,n*sizeof(cl_ulong),pk.data(),0,nullptr,nullptr);if(!err)err=clEnqueueReadBuffer(q,z,CL_TRUE,0,n*sizeof(cl_uint),status.data(),0,nullptr,nullptr);clReleaseMemObject(a);clReleaseMemObject(x);clReleaseMemObject(y);clReleaseMemObject(z);if(err)throw std::runtime_error("OpenCL evaluation failed");for(size_t i=0;i<n;++i)p[b+i].fit={st[i],pk[i],status[i]==0};}
};
#endif
void evaluate(std::vector<Person>&p,const Opt&o,
#ifdef SCOLLATZ_HAS_OPENCL
GPU*g
#else
void*g
#endif
){if(o.backend=="cpu"){cpu(p,0,p.size(),o);return;}
#ifdef SCOLLATZ_HAS_OPENCL
if(o.backend=="opencl"){g->run(p,0,p.size(),o.limit);return;}if(o.backend=="auto"||o.backend=="hybrid"){size_t split=(p.size()+1)/2;std::thread t([&]{cpu(p,split,p.size(),o);});try{g->run(p,0,split,o.limit);}catch(...){t.join();cpu(p,0,split,o);}t.join();return;}
#else
(void)g;if(o.backend=="opencl")throw std::runtime_error("OpenCL not built");
#endif
cpu(p,0,p.size(),o);}
void checkpoint(const Opt&o,const std::vector<Island>&is,size_t gen,U seed){std::ofstream f(o.checkpoint);if(!f)throw std::runtime_error("Cannot write checkpoint");f<<"{\\n\\"version\\":1,\\"generation\\":"<<gen<<",\\"seed\\":"<<seed<<",\\"rng\\":[";for(size_t i=0;i<is.size();++i){if(i)f<<",";std::ostringstream x;x<<is[i].rng;f<<"\\""<<x.str()<<"\\"";}f<<"],\\"populations\\":[";for(size_t i=0;i<is.size();++i){if(i)f<<",";f<<"[";for(size_t j=0;j<is[i].p.size();++j){if(j)f<<",";f<<is[i].p[j].gene;}f<<"]";}f<<"]}\\n";}
bool resume(const std::string&path,std::vector<Island>&is,size_t&gen,U&seed){std::ifstream f(path);if(!f)return false;std::string s((std::istreambuf_iterator<char>(f)),{});std::smatch m;if(!std::regex_search(s,m,std::regex("\\"generation\\"\\s*:\\s*(\\d+)")))return false;gen=std::stoull(m[1]);if(!std::regex_search(s,m,std::regex("\\"seed\\"\\s*:\\s*(\\d+)")))return false;seed=std::stoull(m[1]);if(!std::regex_search(s,m,std::regex("\\"rng\\"\\s*:\\s*\\[([^\\]]*)\\]")))return false;std::vector<std::string>rs;std::regex qr("\\"([^\\"]+)\\"");for(std::sregex_iterator i(m[1].first,m[1].second,qr),e;i!=e;++i)rs.push_back((*i)[1]);if(rs.size()!=is.size())return false;for(size_t i=0;i<is.size();++i){std::istringstream x(rs[i]);if(!(x>>is[i].rng))return false;}if(!std::regex_search(s,m,std::regex("\\"populations\\"\\s*:\\s*\\[(.*)\\]\\s*\\}")))return false;std::string body=m[1];std::regex ar("\\[([^\\]]*)\\]");size_t k=0;for(std::sregex_iterator i(body.begin(),body.end(),ar),e;i!=e;++i,++k){if(k>=is.size())return false;std::istringstream x((*i)[1]);std::string v;while(std::getline(x,v,',')){U n;if(!num(v.c_str(),n))return false;is[k].p.push_back({n,{}});}}if(k!=is.size())return false;for(auto&a:is)if(a.p.empty())return false;return true;}
void outputs(const Opt&o,const std::vector<Rec>&r,U seed){if(!o.csv.empty()){std::ofstream f(o.csv);if(!f)throw std::runtime_error("CSV write failed");f<<"generation,island,candidate,stopping_time,peak,converged,elapsed_seconds\\n";for(auto&x:r)f<<x.gen<<","<<x.island<<","<<x.best.gene<<","<<x.best.fit.steps<<","<<x.best.fit.peak<<","<<(x.best.fit.done?"true":"false")<<","<<x.seconds<<"\\n";}if(!o.json.empty()){std::ofstream f(o.json);if(!f)throw std::runtime_error("JSON write failed");f<<"{\\n\\"seed\\":"<<seed<<",\\"results\\":[";for(size_t i=0;i<r.size();++i){auto&x=r[i];if(i)f<<",";f<<"{\\"generation\\":"<<x.gen<<",\\"island\\":"<<x.island<<",\\"candidate\\":"<<x.best.gene<<",\\"stopping_time\\":"<<x.best.fit.steps<<",\\"peak\\":"<<x.best.fit.peak<<",\\"converged\\":"<<(x.best.fit.done?"true":"false")<<",\\"elapsed_seconds\\":"<<x.seconds<<"}";}f<<"]}\\n";}}
}
int main(int ac,char**av){Opt o;if(!parse(ac,av,o)){std::cerr<<"Invalid arguments; use --help\\n";return 2;}U seed=o.seed?o.seed:(U)std::chrono::high_resolution_clock::now().time_since_epoch().count();std::vector<Island>is(o.islands);for(size_t i=0;i<is.size();++i)is[i].rng.seed(seed+0x9e3779b97f4a7c15ULL*(i+1));size_t first=0;if(!o.resume.empty()){if(!resume(o.resume,is,first,seed)){std::cerr<<"Invalid/incompatible checkpoint\\n";return 2;}}else{size_t base=o.population/o.islands,extra=o.population%o.islands;for(size_t i=0;i<is.size();++i)for(size_t j=0;j<base+(i<extra);++j)is[i].p.push_back({random_u(is[i].rng,o.lo,o.hi),{}});}
#ifdef SCOLLATZ_HAS_OPENCL
std::unique_ptr<GPU>gpu;if(o.backend!="cpu"){try{gpu=std::make_unique<GPU>();}catch(const std::exception&e){if(o.backend=="opencl"){std::cerr<<e.what()<<"\\n";return 3;}std::cerr<<"OpenCL unavailable: "<<e.what()<<"; CPU fallback\\n";o.backend="cpu";}}
#else
if(o.backend=="opencl"){std::cerr<<"OpenCL support not built\\n";return 3;}
#endif
std::vector<Rec>records;Person best{};bool have=false;for(size_t gen=first;gen<o.generations;++gen){auto t0=std::chrono::steady_clock::now();for(size_t k=0;k<is.size();++k){evaluate(is[k].p,o,
#ifdef SCOLLATZ_HAS_OPENCL
gpu.get()
#else
nullptr
#endif
);auto&p=is[k].p;std::sort(p.begin(),p.end(),better);if(!have||better(p[0],best)){best=p[0];have=true;}records.push_back({gen,k,p[0],0});}
if((gen+1)%o.migrate_every==0&&is.size()>1){std::vector<std::vector<Person>>m(is.size());for(size_t k=0;k<is.size();++k)for(size_t n=0;n<std::min(o.migrate_count,is[k].p.size());++n)m[k].push_back(is[k].p[n]);for(size_t k=0;k<is.size();++k){auto&d=is[(k+1)%is.size()].p;for(size_t n=0;n<m[k].size();++n)d[d.size()-1-n]=m[k][n];}}
if(gen+1<o.generations)for(auto&island:is){auto&p=island.p;std::vector<Person>next;size_t elite=std::min(o.elite,p.size()-1);for(size_t n=0;n<elite;++n)next.push_back(p[n]);while(next.size()<p.size()){auto&a=tournament(p,island.rng,o.tournament);auto&b=tournament(p,island.rng,o.tournament);U mask=island.rng();U child=(a.gene&mask)|(b.gene&~mask);next.push_back({mutate(child,o,island.rng),{}});}p.swap(next);}
double sec=std::chrono::duration<double>(std::chrono::steady_clock::now()-t0).count();for(size_t n=records.size()-is.size();n<records.size();++n)records[n].seconds=sec;std::cout<<"generation="<<gen<<" best="<<best.gene<<" stopping_time="<<best.fit.steps<<" backend="<<o.backend<<" islands="<<o.islands<<"\\n";checkpoint(o,is,gen+1,seed);}
outputs(o,records,seed);std::cout<<"best_candidate="<<best.gene<<" stopping_time="<<best.fit.steps<<" peak="<<best.fit.peak<<" seed="<<seed<<"\\n";return best.fit.done?0:1;}
