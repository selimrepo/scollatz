#include <exception>
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
struct Opt{size_t population=512,generations=200,elite=8,tournament=5,islands=4,migrate_every=10,migrate_count=1;unsigned threads=std::max(1u,std::thread::hardware_concurrency());double mutation=.04;bool adaptive_migration=true;U limit=1000000,lo=1,hi=1000000000,seed=0;std::string backend="auto",checkpoint="scollatz.chk",resume,csv="scollatz.csv",json="scollatz.json";};
struct Island{std::vector<Person> p;std::mt19937_64 rng;};
struct Rec{size_t gen,island;Person best;double seconds;};
bool num(const char*s,U&v){try{size_t n=0;auto x=std::stoull(s,&n);if(s[n])return false;v=x;return true;}catch(...){return false;}}
bool sz(const char*s,size_t&v){U x;if(!num(s,x)||x>SIZE_MAX)return false;v=(size_t)x;return true;}
bool dbl(const char*s,double&v){try{size_t n=0;v=std::stod(s,&n);return !s[n]&&v>=0&&v<=1;}catch(...){return false;}}
bool parse(int ac,char**av,Opt&o){for(int i=1;i<ac;++i){std::string a=av[i];if(a=="--help"||a=="-h"){std::cout<<"scollatz-ga --backend auto|cpu|opencl|hybrid --islands N --population N --generations N --migration-interval N --migration-count N --no-adaptive-migration --checkpoint FILE --resume FILE --csv FILE --json FILE [--threads N --min N --max N --seed N]\n";std::exit(0);}if(i+1>=ac)return false;const char*v=av[++i];
if(a=="--no-adaptive-migration"){o.adaptive_migration=false;continue;}if(a=="--population"){if(!sz(v,o.population))return false;}else if(a=="--generations"){if(!sz(v,o.generations))return false;}else if(a=="--elite"){if(!sz(v,o.elite))return false;}else if(a=="--tournament"){if(!sz(v,o.tournament))return false;}else if(a=="--islands"){if(!sz(v,o.islands))return false;}else if(a=="--migration-interval"){if(!sz(v,o.migrate_every))return false;}else if(a=="--migration-count"){if(!sz(v,o.migrate_count))return false;}else if(a=="--threads"){U x;if(!num(v,x)||!x||x>4096)return false;o.threads=(unsigned)x;}else if(a=="--mutation"){if(!dbl(v,o.mutation))return false;}else if(a=="--max-steps"){if(!num(v,o.limit))return false;}else if(a=="--min"){if(!num(v,o.lo))return false;}else if(a=="--max"){if(!num(v,o.hi))return false;}else if(a=="--seed"){if(!num(v,o.seed))return false;}else if(a=="--backend")o.backend=v;else if(a=="--checkpoint")o.checkpoint=v;else if(a=="--resume")o.resume=v;else if(a=="--csv")o.csv=v;else if(a=="--json")o.json=v;else return false;}
return o.population>=8&&o.generations&&o.islands&&o.islands<=o.population&&o.elite&&o.elite<o.population/o.islands&&o.tournament&&o.limit&&o.lo&&o.hi>=o.lo&&o.migrate_every&&o.migrate_count&&(o.backend=="auto"||o.backend=="cpu"||o.backend=="opencl"||o.backend=="hybrid");}
Fit fitness(U n,U lim){U s=0,p=n;while(n!=1&&s<lim){if(!(n&1))n>>=1;else{if(n>(UINT64_MAX-1)/3)return{s,p,false};n=n*3+1;}++s;p=std::max(p,n);}return{s,p,n==1};}
bool better(const Person&a,const Person&b){if(a.fit.done!=b.fit.done)return a.fit.done;if(a.fit.steps!=b.fit.steps)return a.fit.steps>b.fit.steps;if(a.fit.peak!=b.fit.peak)return a.fit.peak>b.fit.peak;return a.gene<b.gene;}
U random_u(std::mt19937_64&r,U a,U b){return std::uniform_int_distribution<U>(a,b)(r);}
U mutate(U x,const Opt&o,std::mt19937_64&r){std::bernoulli_distribution f(o.mutation);for(unsigned b=0;b<64;++b)if(f(r))x^=U(1)<<b;if(f(r)){U d=random_u(r,0,std::max<U>(1,(o.hi-o.lo)/32));if(f(r))x=x>d?x-d:o.lo;else if(UINT64_MAX-x>=d)x+=d;}return std::clamp(x,o.lo,o.hi);}
const Person& tournament(const std::vector<Person>&p,std::mt19937_64&r,size_t n){size_t b=random_u(r,0,p.size()-1);while(--n){size_t j=random_u(r,0,p.size()-1);if(better(p[j],p[b]))b=j;}return p[b];}
void cpu(std::vector<Person>&p,size_t b,size_t e,const Opt&o){std::atomic<size_t>next{b};std::vector<std::thread>w;unsigned n=std::min<unsigned>(o.threads,(unsigned)(e-b));for(unsigned t=0;t<n;++t)w.emplace_back([&]{for(;;){size_t i=next++;if(i>=e)break;p[i].fit=fitness(p[i].gene,o.limit);}});for(auto&t:w)t.join();}
#ifdef SCOLLATZ_HAS_OPENCL
class GPU {
    cl_context context_=nullptr;
    cl_command_queue queue_=nullptr;
    cl_program program_=nullptr;
    cl_kernel kernel_=nullptr;
    cl_mem candidates_=nullptr,steps_=nullptr,peaks_=nullptr,status_=nullptr;
    size_t capacity_=0;
    std::vector<cl_ulong> host_candidates_,host_steps_,host_peaks_;
    std::vector<cl_uint> host_status_;
    cl_device_id device_=nullptr;
    void release() noexcept {
        if(status_)clReleaseMemObject(status_);
        if(peaks_)clReleaseMemObject(peaks_);
        if(steps_)clReleaseMemObject(steps_);
        if(candidates_)clReleaseMemObject(candidates_);
        if(kernel_)clReleaseKernel(kernel_);
        if(program_)clReleaseProgram(program_);
        if(queue_)clReleaseCommandQueue(queue_);
        if(context_)clReleaseContext(context_);
        status_=peaks_=steps_=candidates_=nullptr;
        kernel_=nullptr;program_=nullptr;queue_=nullptr;context_=nullptr;
    }
    void check(cl_int e,const char*what){if(e!=CL_SUCCESS)throw std::runtime_error(std::string(what)+" (OpenCL "+std::to_string(e)+")");}
public:
    explicit GPU(size_t capacity):capacity_(capacity),host_candidates_(capacity),host_steps_(capacity),host_peaks_(capacity),host_status_(capacity) {
        try {
            cl_uint np=0;check(clGetPlatformIDs(0,nullptr,&np),"OpenCL platform query");
            if(!np)throw std::runtime_error("No OpenCL platform");
            std::vector<cl_platform_id> platforms(np);check(clGetPlatformIDs(np,platforms.data(),nullptr),"OpenCL platform enumeration");
            const cl_device_type types[]={static_cast<cl_device_type>(CL_DEVICE_TYPE_GPU),CL_DEVICE_TYPE_ALL};
            for(cl_device_type type:types){for(auto platform:platforms){if(clGetDeviceIDs(platform,type,1,&device_,nullptr)==CL_SUCCESS)break;device_=nullptr;}if(device_)break;}
            if(!device_)throw std::runtime_error("No OpenCL device");
            cl_int e=CL_SUCCESS;context_=clCreateContext(nullptr,1,&device_,nullptr,nullptr,&e);check(e,"OpenCL context creation");
            queue_=clCreateCommandQueue(context_,device_,0,&e);check(e,"OpenCL command queue creation");
            std::ifstream file(SCOLLATZ_GA_KERNEL_PATH);if(!file)throw std::runtime_error("Missing GA kernel");
            std::ostringstream source;source<<file.rdbuf();const std::string code=source.str();const char*ptr=code.c_str();const size_t length=code.size();
            program_=clCreateProgramWithSource(context_,1,&ptr,&length,&e);check(e,"OpenCL program creation");
            e=clBuildProgram(program_,1,&device_,nullptr,nullptr,nullptr);
            if(e!=CL_SUCCESS){size_t n=0;clGetProgramBuildInfo(program_,device_,CL_PROGRAM_BUILD_LOG,0,nullptr,&n);std::string log(n,' ');if(n)clGetProgramBuildInfo(program_,device_,CL_PROGRAM_BUILD_LOG,n,log.data(),nullptr);throw std::runtime_error("OpenCL build failed: "+log);}
            kernel_=clCreateKernel(program_,"collatz_candidates",&e);check(e,"OpenCL kernel creation");
            candidates_=clCreateBuffer(context_,CL_MEM_READ_ONLY,capacity_*sizeof(cl_ulong),nullptr,&e);check(e,"Candidate buffer allocation");
            steps_=clCreateBuffer(context_,CL_MEM_WRITE_ONLY,capacity_*sizeof(cl_ulong),nullptr,&e);check(e,"Steps buffer allocation");
            peaks_=clCreateBuffer(context_,CL_MEM_WRITE_ONLY,capacity_*sizeof(cl_ulong),nullptr,&e);check(e,"Peaks buffer allocation");
            status_=clCreateBuffer(context_,CL_MEM_WRITE_ONLY,capacity_*sizeof(cl_uint),nullptr,&e);check(e,"Status buffer allocation");
            char name[256]{};clGetDeviceInfo(device_,CL_DEVICE_NAME,sizeof(name),name,nullptr);
            std::cout<<"OpenCL island executor: "<<name<<", persistent capacity="<<capacity_<<"\n";
        } catch(...) {release();throw;}
    }
    GPU(const GPU&)=delete;GPU&operator=(const GPU&)=delete;
    ~GPU(){release();}
    void run(std::vector<Person>&people,U limit) {
        const size_t n=people.size();if(!n)return;
        if(n>capacity_)throw std::runtime_error("Island population exceeds persistent OpenCL buffer capacity");
        for(size_t i=0;i<n;++i)host_candidates_[i]=static_cast<cl_ulong>(people[i].gene);
        cl_int e=clEnqueueWriteBuffer(queue_,candidates_,CL_FALSE,0,n*sizeof(cl_ulong),host_candidates_.data(),0,nullptr,nullptr);check(e,"Candidate upload");
        const cl_ulong count=static_cast<cl_ulong>(n),max_steps=static_cast<cl_ulong>(limit);
        check(clSetKernelArg(kernel_,0,sizeof(candidates_),&candidates_),"Set candidates arg");
        check(clSetKernelArg(kernel_,1,sizeof(count),&count),"Set count arg");
        check(clSetKernelArg(kernel_,2,sizeof(steps_),&steps_),"Set steps arg");
        check(clSetKernelArg(kernel_,3,sizeof(peaks_),&peaks_),"Set peaks arg");
        check(clSetKernelArg(kernel_,4,sizeof(status_),&status_),"Set status arg");
        check(clSetKernelArg(kernel_,5,sizeof(max_steps),&max_steps),"Set limit arg");
        const size_t global=n;
        check(clEnqueueNDRangeKernel(queue_,kernel_,1,nullptr,&global,nullptr,0,nullptr,nullptr),"Launch Collatz fitness kernel");
        check(clEnqueueReadBuffer(queue_,steps_,CL_TRUE,0,n*sizeof(cl_ulong),host_steps_.data(),0,nullptr,nullptr),"Read stopping times");
        check(clEnqueueReadBuffer(queue_,peaks_,CL_TRUE,0,n*sizeof(cl_ulong),host_peaks_.data(),0,nullptr,nullptr),"Read trajectory peaks");
        check(clEnqueueReadBuffer(queue_,status_,CL_TRUE,0,n*sizeof(cl_uint),host_status_.data(),0,nullptr,nullptr),"Read fitness status");
        for(size_t i=0;i<n;++i)people[i].fit={static_cast<U>(host_steps_[i]),static_cast<U>(host_peaks_[i]),host_status_[i]==0};
    }
};
#endif

void evaluate(std::vector<Person>&p,const Opt&o,
#ifdef SCOLLATZ_HAS_OPENCL
GPU*g
#else
void*g
#endif
){
    if(o.backend=="cpu"){cpu(p,0,p.size(),o);return;}
#ifdef SCOLLATZ_HAS_OPENCL
    if(!g)throw std::runtime_error("Missing island OpenCL executor");
    if(o.backend=="opencl"){g->run(p,o.limit);return;}
    if(o.backend=="auto"||o.backend=="hybrid"){
        const size_t split=(p.size()+1)/2;
        std::thread worker([&]{cpu(p,split,p.size(),o);});
        try{std::vector<Person> gpu_people(p.begin(),p.begin()+static_cast<std::ptrdiff_t>(split));g->run(gpu_people,o.limit);std::copy(gpu_people.begin(),gpu_people.end(),p.begin());}
        catch(...){worker.join();cpu(p,0,split,o);return;}
        worker.join();return;
    }
#else
    (void)g;if(o.backend=="opencl")throw std::runtime_error("OpenCL not built");
#endif
    cpu(p,0,p.size(),o);
}
void checkpoint(const Opt&o,const std::vector<Island>&is,size_t gen,U seed){std::ofstream f(o.checkpoint);if(!f)throw std::runtime_error("Cannot write checkpoint");f<<"{\n\"version\":1,\"generation\":"<<gen<<",\"seed\":"<<seed<<",\"rng\":[";for(size_t i=0;i<is.size();++i){if(i)f<<",";std::ostringstream x;x<<is[i].rng;f<<"\""<<x.str()<<"\"";}f<<"],\"populations\":[";for(size_t i=0;i<is.size();++i){if(i)f<<",";f<<"[";for(size_t j=0;j<is[i].p.size();++j){if(j)f<<",";f<<is[i].p[j].gene;}f<<"]";}f<<"]}\n";}
bool resume(const std::string&path,std::vector<Island>&is,size_t&gen,U&seed){std::ifstream f(path);if(!f)return false;std::string s((std::istreambuf_iterator<char>(f)),{});std::smatch m;if(!std::regex_search(s,m,std::regex("\"generation\"\\s*:\\s*(\\d+)")))return false;gen=std::stoull(m[1]);if(!std::regex_search(s,m,std::regex("\"seed\"\\s*:\\s*(\\d+)")))return false;seed=std::stoull(m[1]);if(!std::regex_search(s,m,std::regex("\"rng\"\\s*:\\s*\\[([^\\]]*)\\]")))return false;std::vector<std::string>rs;std::regex qr("\"([^\"]+)\"");for(std::sregex_iterator i(m[1].first,m[1].second,qr),e;i!=e;++i)rs.push_back((*i)[1]);if(rs.size()!=is.size())return false;for(size_t i=0;i<is.size();++i){std::istringstream x(rs[i]);if(!(x>>is[i].rng))return false;}if(!std::regex_search(s,m,std::regex("\"populations\"\\s*:\\s*\\[(.*)\\]\\s*\\}")))return false;std::string body=m[1];std::regex ar("\\[([^\\]]*)\\]");size_t k=0;for(std::sregex_iterator i(body.begin(),body.end(),ar),e;i!=e;++i,++k){if(k>=is.size())return false;std::istringstream x((*i)[1]);std::string v;while(std::getline(x,v,',')){U n;if(!num(v.c_str(),n))return false;is[k].p.push_back({n,{}});}}if(k!=is.size())return false;for(auto&a:is)if(a.p.empty())return false;return true;}
void outputs(const Opt&o,const std::vector<Rec>&r,U seed){if(!o.csv.empty()){std::ofstream f(o.csv);if(!f)throw std::runtime_error("CSV write failed");f<<"generation,island,candidate,stopping_time,peak,converged,elapsed_seconds\n";for(auto&x:r)f<<x.gen<<","<<x.island<<","<<x.best.gene<<","<<x.best.fit.steps<<","<<x.best.fit.peak<<","<<(x.best.fit.done?"true":"false")<<","<<x.seconds<<"\n";}if(!o.json.empty()){std::ofstream f(o.json);if(!f)throw std::runtime_error("JSON write failed");f<<"{\n\"seed\":"<<seed<<",\"results\":[";for(size_t i=0;i<r.size();++i){auto&x=r[i];if(i)f<<",";f<<"{\"generation\":"<<x.gen<<",\"island\":"<<x.island<<",\"candidate\":"<<x.best.gene<<",\"stopping_time\":"<<x.best.fit.steps<<",\"peak\":"<<x.best.fit.peak<<",\"converged\":"<<(x.best.fit.done?"true":"false")<<",\"elapsed_seconds\":"<<x.seconds<<"}";}f<<"]}\n";}}
}
int main(int ac,char**av){Opt o;if(!parse(ac,av,o)){std::cerr<<"Invalid arguments; use --help\n";return 2;}U seed=o.seed?o.seed:(U)std::chrono::high_resolution_clock::now().time_since_epoch().count();std::vector<Island>is(o.islands);for(size_t i=0;i<is.size();++i)is[i].rng.seed(seed+0x9e3779b97f4a7c15ULL*(i+1));size_t first=0;if(!o.resume.empty()){if(!resume(o.resume,is,first,seed)){std::cerr<<"Invalid/incompatible checkpoint\n";return 2;}}else{size_t base=o.population/o.islands,extra=o.population%o.islands;for(size_t i=0;i<is.size();++i)for(size_t j=0;j<base+(i<extra);++j)is[i].p.push_back({random_u(is[i].rng,o.lo,o.hi),{}});}
#ifdef SCOLLATZ_HAS_OPENCL
std::vector<std::unique_ptr<GPU>>gpus(is.size());
if(o.backend!="cpu"){try{for(size_t k=0;k<is.size();++k)gpus[k]=std::make_unique<GPU>(is[k].p.size());}catch(const std::exception&e){if(o.backend=="opencl"){std::cerr<<e.what()<<"\n";return 3;}std::cerr<<"OpenCL unavailable: "<<e.what()<<"; CPU fallback\n";o.backend="cpu";for(auto&g:gpus)g.reset();}}
#else
if(o.backend=="opencl"){std::cerr<<"OpenCL support not built\n";return 3;}
#endif
std::vector<Rec>records;Person best{};bool have=false;for(size_t gen=first;gen<o.generations;++gen){auto t0=std::chrono::steady_clock::now();
std::vector<std::thread>island_workers;std::vector<std::exception_ptr>island_errors(is.size());std::vector<Opt>island_options(is.size(),o);
for(size_t k=0;k<is.size();++k){island_options[k].threads=std::max(1u,o.threads/static_cast<unsigned>(is.size()));island_workers.emplace_back([&,k]{try{evaluate(is[k].p,island_options[k],
#ifdef SCOLLATZ_HAS_OPENCL
gpus[k].get()
#else
nullptr
#endif
);std::sort(is[k].p.begin(),is[k].p.end(),better);}catch(...){island_errors[k]=std::current_exception();}});}
for(auto&worker:island_workers)worker.join();for(const auto&error:island_errors)if(error)std::rethrow_exception(error);
for(size_t k=0;k<is.size();++k){auto&p=is[k].p;if(!have||better(p[0],best)){best=p[0];have=true;}records.push_back({gen,k,p[0],0});}
{
size_t migration_interval=o.migrate_every,migration_count=o.migrate_count;
double diversity=0.0;
if(o.adaptive_migration){
    for(const auto& island:is){std::vector<U>genes;genes.reserve(island.p.size());for(const auto&person:island.p)genes.push_back(person.gene);std::sort(genes.begin(),genes.end());const auto unique=std::unique(genes.begin(),genes.end())-genes.begin();diversity+=island.p.empty()?0.0:static_cast<double>(unique)/static_cast<double>(island.p.size());}
    diversity/=static_cast<double>(is.size());
    if(diversity<0.25){migration_interval=std::max<size_t>(1,o.migrate_every/2);migration_count=std::min(o.migrate_count*2,std::max<size_t>(1,o.population/o.islands/8));}
    else if(diversity>0.70){migration_interval=std::min<size_t>(SIZE_MAX/2,o.migrate_every)*2;migration_count=std::max<size_t>(1,o.migrate_count/2);}
}
if(is.size()>1&&(gen+1)%migration_interval==0){
    std::vector<std::vector<Person>>migrants(is.size());
    for(size_t k=0;k<is.size();++k){const size_t n=std::min({migration_count,is[k].p.size(),is[(k+1)%is.size()].p.size()});for(size_t j=0;j<n;++j)migrants[k].push_back(is[k].p[j]);}
    for(size_t k=0;k<is.size();++k){auto&destination=is[(k+1)%is.size()].p;for(size_t j=0;j<migrants[k].size();++j)destination[destination.size()-1-j]=migrants[k][j];}
    std::cout<<"migration generation="<<gen<<" diversity="<<diversity<<" interval="<<migration_interval<<" migrants_per_island="<<migration_count<<"\n";
}
}
if(gen+1<o.generations)for(auto&island:is){auto&p=island.p;std::vector<Person>next;size_t elite=std::min(o.elite,p.size()-1);for(size_t n=0;n<elite;++n)next.push_back(p[n]);while(next.size()<p.size()){auto&a=tournament(p,island.rng,o.tournament);auto&b=tournament(p,island.rng,o.tournament);U mask=island.rng();U child=(a.gene&mask)|(b.gene&~mask);next.push_back({mutate(child,o,island.rng),{}});}p.swap(next);}
double sec=std::chrono::duration<double>(std::chrono::steady_clock::now()-t0).count();for(size_t n=records.size()-is.size();n<records.size();++n)records[n].seconds=sec;std::cout<<"generation="<<gen<<" best="<<best.gene<<" stopping_time="<<best.fit.steps<<" backend="<<o.backend<<" islands="<<o.islands<<"\n";checkpoint(o,is,gen+1,seed);}
outputs(o,records,seed);std::cout<<"best_candidate="<<best.gene<<" stopping_time="<<best.fit.steps<<" peak="<<best.fit.peak<<" seed="<<seed<<"\n";return best.fit.done?0:1;}
