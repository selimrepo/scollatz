#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <mutex>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

#ifdef SCOLLATZ_HAS_OPENCL
#define CL_TARGET_OPENCL_VERSION 120
#include <CL/cl.h>
#endif

namespace {
using u64 = std::uint64_t;

struct Options {
    u64 start = 1;
    u64 end = 1'000'000;
    u64 max_steps = 10'000'000;
    u64 chunk = 4096;
    unsigned threads = std::max(1u, std::thread::hardware_concurrency());
    std::string backend = "auto";
};

struct Result {
    u64 tested = 0, completed = 0, overflow = 0, capped = 0;
    u64 max_steps = 0, max_steps_start = 0, max_peak = 0, max_peak_start = 0;
};

bool number(const char* s, u64& out) {
    try {
        std::size_t pos = 0;
        const auto value = std::stoull(s, &pos);
        if (s[pos] != '\0') return false;
        out = static_cast<u64>(value);
        return true;
    } catch (...) { return false; }
}

bool parse(int argc, char** argv, Options& o) {
    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        auto value = [&](u64& target) {
            return i + 1 < argc && number(argv[++i], target);
        };
        if (arg == "--start") { if (!value(o.start)) return false; }
        else if (arg == "--end") { if (!value(o.end)) return false; }
        else if (arg == "--max-steps") { if (!value(o.max_steps)) return false; }
        else if (arg == "--chunk") { if (!value(o.chunk)) return false; }
        else if (arg == "--threads") {
            u64 n{}; if (!value(n) || n == 0 || n > 4096) return false;
            o.threads = static_cast<unsigned>(n);
        } else if (arg == "--backend") {
            if (i + 1 >= argc) return false;
            o.backend = argv[++i];
        } else if (arg == "--help" || arg == "-h") {
            std::cout << "scollatz --start N --end N [--backend auto|cpu|opencl] "
                         "[--threads N] [--chunk N] [--max-steps N]\n"
                         "Scans the inclusive range and reports stopping-time and peak records.\n";
            std::exit(0);
        } else return false;
    }
    return o.start >= 1 && o.end >= o.start && o.max_steps > 0 &&
           o.chunk > 0 && o.backend != "" &&
           (o.backend == "auto" || o.backend == "cpu" || o.backend == "opencl");
}

struct Sample { u64 steps, peak; unsigned status; };
// status: 0 completed, 1 uint64 overflow, 2 step limit reached
Sample evaluate(u64 n, u64 limit) {
    const u64 initial = n;
    u64 steps = 0, peak = n;
    while (n != 1 && steps < limit) {
        if ((n & 1U) == 0) n >>= 1U;
        else {
            if (n > (std::numeric_limits<u64>::max() - 1U) / 3U)
                return {steps, peak, 1};
            n = n * 3U + 1U;
        }
        ++steps;
        peak = std::max(peak, n);
    }
    (void)initial;
    return {steps, peak, n == 1 ? 0U : 2U};
}

void merge(Result& r, u64 start, const Sample& s) {
    ++r.tested;
    if (s.status == 0) {
        ++r.completed;
        if (s.steps > r.max_steps) { r.max_steps = s.steps; r.max_steps_start = start; }
        if (s.peak > r.max_peak) { r.max_peak = s.peak; r.max_peak_start = start; }
    } else if (s.status == 1) ++r.overflow;
    else ++r.capped;
}

Result run_cpu(const Options& o) {
    std::atomic<u64> next{o.start};
    std::atomic<bool> exhausted{false};
    std::mutex result_mutex;
    Result total;
    const unsigned count = std::min<unsigned>(o.threads,
        static_cast<unsigned>(std::min<u64>(o.end - o.start + 1, 4096)));
    std::vector<std::thread> workers;
    workers.reserve(count);
    for (unsigned t = 0; t < count; ++t) {
        workers.emplace_back([&] {
            Result local;
            while (!exhausted.load(std::memory_order_relaxed)) {
                const u64 begin = next.fetch_add(o.chunk, std::memory_order_relaxed);
                if (begin > o.end) { exhausted.store(true, std::memory_order_relaxed); break; }
                const u64 remaining = o.end - begin;
                const u64 length = std::min(o.chunk, remaining + 1);
                for (u64 offset = 0; offset < length; ++offset) {
                    const u64 n = begin + offset;
                    merge(local, n, evaluate(n, o.max_steps));
                }
            }
            std::lock_guard lock(result_mutex);
            total.tested += local.tested; total.completed += local.completed;
            total.overflow += local.overflow; total.capped += local.capped;
            if (local.max_steps > total.max_steps) {
                total.max_steps = local.max_steps; total.max_steps_start = local.max_steps_start;
            }
            if (local.max_peak > total.max_peak) {
                total.max_peak = local.max_peak; total.max_peak_start = local.max_peak_start;
            }
        });
    }
    for (auto& worker : workers) worker.join();
    return total;
}

#ifdef SCOLLATZ_HAS_OPENCL
std::string read_kernel() {
    std::ifstream in(SCOLLATZ_KERNEL_PATH);
    if (!in) throw std::runtime_error("Cannot read OpenCL kernel: " SCOLLATZ_KERNEL_PATH);
    std::ostringstream ss; ss << in.rdbuf(); return ss.str();
}

Result run_opencl(const Options& o) {
    cl_uint platform_count = 0;
    cl_int err = clGetPlatformIDs(0, nullptr, &platform_count);
    if (err != CL_SUCCESS || platform_count == 0) throw std::runtime_error("No OpenCL platform found");
    std::vector<cl_platform_id> platforms(platform_count);
    clGetPlatformIDs(platform_count, platforms.data(), nullptr);
    cl_device_id device = nullptr;
    for (auto p : platforms) {
        if (clGetDeviceIDs(p, CL_DEVICE_TYPE_GPU, 1, &device, nullptr) == CL_SUCCESS) break;
        device = nullptr;
    }
    if (!device) throw std::runtime_error("No OpenCL GPU device found");
    char device_name[256]{};
    clGetDeviceInfo(device, CL_DEVICE_NAME, sizeof(device_name), device_name, nullptr);
    cl_context context = clCreateContext(nullptr, 1, &device, nullptr, nullptr, &err);
    if (err != CL_SUCCESS) throw std::runtime_error("clCreateContext failed");
    cl_command_queue queue = clCreateCommandQueue(context, device, 0, &err);
    if (err != CL_SUCCESS) { clReleaseContext(context); throw std::runtime_error("clCreateCommandQueue failed"); }
    const std::string source = read_kernel();
    const char* src = source.c_str(); const std::size_t src_len = source.size();
    cl_program program = clCreateProgramWithSource(context, 1, &src, &src_len, &err);
    if (err == CL_SUCCESS) err = clBuildProgram(program, 1, &device, nullptr, nullptr, nullptr);
    if (err != CL_SUCCESS) {
        std::size_t log_size = 0; clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, 0, nullptr, &log_size);
        std::string log(log_size, '\0');
        if (log_size) clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, log_size, log.data(), nullptr);
        clReleaseProgram(program); clReleaseCommandQueue(queue); clReleaseContext(context);
        throw std::runtime_error("OpenCL build failed: " + log);
    }
    cl_kernel kernel = clCreateKernel(program, "collatz_batch", &err);
    if (err != CL_SUCCESS) throw std::runtime_error("clCreateKernel failed");
    Result total;
    const u64 capacity = std::min<u64>(o.chunk, o.end - o.start + 1);
    std::vector<cl_ulong> steps(static_cast<std::size_t>(capacity)), peaks(static_cast<std::size_t>(capacity));
    std::vector<cl_uint> statuses(static_cast<std::size_t>(capacity));
    cl_mem steps_buf = clCreateBuffer(context, CL_MEM_WRITE_ONLY, steps.size()*sizeof(cl_ulong), nullptr, &err);
    cl_mem peaks_buf = clCreateBuffer(context, CL_MEM_WRITE_ONLY, peaks.size()*sizeof(cl_ulong), nullptr, &err);
    cl_mem status_buf = clCreateBuffer(context, CL_MEM_WRITE_ONLY, statuses.size()*sizeof(cl_uint), nullptr, &err);
    if (err != CL_SUCCESS) throw std::runtime_error("OpenCL buffer allocation failed");
    clSetKernelArg(kernel, 2, sizeof(cl_mem), &steps_buf);
    clSetKernelArg(kernel, 3, sizeof(cl_mem), &peaks_buf);
    clSetKernelArg(kernel, 4, sizeof(cl_mem), &status_buf);
    clSetKernelArg(kernel, 5, sizeof(cl_ulong), &o.max_steps);
    std::cout << "OpenCL device: " << device_name << "\n";
    for (u64 begin = o.start; begin <= o.end;) {
        const u64 count = std::min<u64>(capacity, o.end - begin + 1);
        const cl_ulong first = begin, amount = count;
        clSetKernelArg(kernel, 0, sizeof(first), &first);
        clSetKernelArg(kernel, 1, sizeof(amount), &amount);
        const std::size_t global = static_cast<std::size_t>(count);
        err = clEnqueueNDRangeKernel(queue, kernel, 1, nullptr, &global, nullptr, 0, nullptr, nullptr);
        if (err != CL_SUCCESS) throw std::runtime_error("OpenCL kernel dispatch failed");
        clEnqueueReadBuffer(queue, steps_buf, CL_TRUE, 0, global*sizeof(cl_ulong), steps.data(), 0, nullptr, nullptr);
        clEnqueueReadBuffer(queue, peaks_buf, CL_TRUE, 0, global*sizeof(cl_ulong), peaks.data(), 0, nullptr, nullptr);
        clEnqueueReadBuffer(queue, status_buf, CL_TRUE, 0, global*sizeof(cl_uint), statuses.data(), 0, nullptr, nullptr);
        for (u64 i = 0; i < count; ++i)
            merge(total, begin+i, {steps[static_cast<std::size_t>(i)], peaks[static_cast<std::size_t>(i)],
                                   statuses[static_cast<std::size_t>(i)]});
        if (count == o.end - begin + 1) break;
        begin += count;
    }
    clReleaseMemObject(steps_buf); clReleaseMemObject(peaks_buf); clReleaseMemObject(status_buf);
    clReleaseKernel(kernel); clReleaseProgram(program); clReleaseCommandQueue(queue); clReleaseContext(context);
    return total;
}
#endif

void print(const Result& r, double seconds, const std::string& backend) {
    std::cout << "Backend: " << backend << "\n"
              << "Tested: " << r.tested << " | completed: " << r.completed
              << " | overflow: " << r.overflow << " | step-limit: " << r.capped << "\n"
              << "Longest stopping time: " << r.max_steps << " (start " << r.max_steps_start << ")\n"
              << "Highest peak: " << r.max_peak << " (start " << r.max_peak_start << ")\n"
              << std::fixed << std::setprecision(3) << "Elapsed: " << seconds << " s\n";
    if (seconds > 0) std::cout << "Throughput: " << std::fixed << std::setprecision(0)
                               << static_cast<double>(r.tested)/seconds << " starts/s\n";
}
} // namespace

int main(int argc, char** argv) {
    Options options;
    if (!parse(argc, argv, options)) {
        std::cerr << "Invalid arguments. Use --help for usage. Range must be positive and inclusive.\n";
        return 2;
    }
    const auto begin = std::chrono::steady_clock::now();
    Result result;
    std::string backend = options.backend;
    try {
#ifdef SCOLLATZ_HAS_OPENCL
        if (backend == "opencl" || backend == "auto") {
            try { result = run_opencl(options); backend = "OpenCL GPU"; }
            catch (const std::exception& e) {
                if (options.backend == "opencl") throw;
                std::cerr << "OpenCL unavailable (" << e.what() << "); falling back to CPU.\n";
                result = run_cpu(options); backend = "CPU (" + std::to_string(options.threads) + " threads)";
            }
        } else
#else
        if (backend == "opencl") throw std::runtime_error("This build has no OpenCL support");
#endif
        { result = run_cpu(options); backend = "CPU (" + std::to_string(options.threads) + " threads)"; }
        const double elapsed = std::chrono::duration<double>(std::chrono::steady_clock::now()-begin).count();
        print(result, elapsed, backend);
        return result.capped == 0 && result.overflow == 0 ? 0 : 1;
    } catch (const std::exception& e) {
        std::cerr << "Fatal: " << e.what() << "\n"; return 3;
    }
}
