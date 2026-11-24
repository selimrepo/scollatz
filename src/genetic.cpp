#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <limits>
#include <mutex>
#include <random>
#include <string>
#include <thread>
#include <vector>

namespace {
using u64 = std::uint64_t;

struct Options {
    std::size_t population = 512;
    std::size_t generations = 200;
    std::size_t elite = 16;
    std::size_t tournament = 5;
    unsigned threads = std::max(1u, std::thread::hardware_concurrency());
    double mutation = 0.04;
    u64 max_steps = 1'000'000;
    u64 min_value = 1;
    u64 max_value = 1'000'000'000;
    u64 seed = 0;
};

struct Fitness {
    u64 steps = 0;
    u64 peak = 0;
    bool complete = false;
};

struct Individual {
    u64 genome = 1;
    Fitness fitness{};
};

bool parse_u64(const char* text, u64& out) {
    try {
        std::size_t pos = 0;
        const auto value = std::stoull(text, &pos);
        if (text[pos] != '\0') return false;
        out = static_cast<u64>(value);
        return true;
    } catch (...) { return false; }
}

bool parse_size(const char* text, std::size_t& out) {
    u64 value{};
    if (!parse_u64(text, value) || value > std::numeric_limits<std::size_t>::max()) return false;
    out = static_cast<std::size_t>(value);
    return true;
}

bool parse_double(const char* text, double& out) {
    try {
        std::size_t pos = 0;
        out = std::stod(text, &pos);
        return text[pos] == '\0' && out >= 0.0 && out <= 1.0;
    } catch (...) { return false; }
}

bool parse(int argc, char** argv, Options& o) {
    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        if (i + 1 >= argc) return false;
        const char* value = argv[++i];
        if (arg == "--population") { if (!parse_size(value, o.population)) return false; }
        else if (arg == "--generations") { if (!parse_size(value, o.generations)) return false; }
        else if (arg == "--elite") { if (!parse_size(value, o.elite)) return false; }
        else if (arg == "--tournament") { if (!parse_size(value, o.tournament)) return false; }
        else if (arg == "--threads") {
            u64 n{}; if (!parse_u64(value, n) || n == 0 || n > 4096) return false;
            o.threads = static_cast<unsigned>(n);
        } else if (arg == "--mutation") { if (!parse_double(value, o.mutation)) return false; }
        else if (arg == "--max-steps") { if (!parse_u64(value, o.max_steps)) return false; }
        else if (arg == "--min") { if (!parse_u64(value, o.min_value)) return false; }
        else if (arg == "--max") { if (!parse_u64(value, o.max_value)) return false; }
        else if (arg == "--seed") { if (!parse_u64(value, o.seed)) return false; }
        else if (arg == "--help" || arg == "-h") {
            std::cout << "scollatz-ga --population N --generations N --min N --max N "
                         "--threads N --mutation 0..1 --elite N --tournament N --seed N\n";
            std::exit(0);
        } else return false;
    }
    return o.population >= 8 && o.generations > 0 && o.elite > 0 &&
           o.elite < o.population && o.tournament > 0 &&
           o.tournament <= o.population && o.max_steps > 0 &&
           o.min_value >= 1 && o.max_value >= o.min_value;
}

Fitness evaluate(u64 start, u64 limit) {
    u64 n = start, steps = 0, peak = start;
    while (n != 1 && steps < limit) {
        if ((n & 1U) == 0) n >>= 1U;
        else {
            if (n > (std::numeric_limits<u64>::max() - 1U) / 3U)
                return {steps, peak, false};
            n = n * 3U + 1U;
        }
        ++steps;
        peak = std::max(peak, n);
    }
    return {steps, peak, n == 1};
}

bool better(const Individual& a, const Individual& b) {
    if (a.fitness.complete != b.fitness.complete) return a.fitness.complete;
    if (a.fitness.steps != b.fitness.steps) return a.fitness.steps > b.fitness.steps;
    if (a.fitness.peak != b.fitness.peak) return a.fitness.peak > b.fitness.peak;
    return a.genome < b.genome;
}

void evaluate_population(std::vector<Individual>& population, const Options& o) {
    std::atomic<std::size_t> next{0};
    const unsigned count = std::min<unsigned>(o.threads, static_cast<unsigned>(population.size()));
    std::vector<std::thread> workers;
    workers.reserve(count);
    for (unsigned t = 0; t < count; ++t) {
        workers.emplace_back([&] {
            for (;;) {
                const std::size_t i = next.fetch_add(1, std::memory_order_relaxed);
                if (i >= population.size()) break;
                population[i].fitness = evaluate(population[i].genome, o.max_steps);
            }
        });
    }
    for (auto& worker : workers) worker.join();
}

u64 random_value(std::mt19937_64& rng, u64 low, u64 high) {
    std::uniform_int_distribution<u64> dist(low, high);
    return dist(rng);
}

u64 clamp_genome(u64 value, const Options& o) {
    return std::clamp(value, o.min_value, o.max_value);
}

const Individual& tournament_pick(const std::vector<Individual>& population,
                                   std::mt19937_64& rng, std::size_t size) {
    std::uniform_int_distribution<std::size_t> pick(0, population.size() - 1);
    std::size_t best = pick(rng);
    for (std::size_t i = 1; i < size; ++i) {
        const std::size_t candidate = pick(rng);
        if (better(population[candidate], population[best])) best = candidate;
    }
    return population[best];
}

u64 crossover(u64 a, u64 b, std::mt19937_64& rng) {
    const u64 mask = rng();
    return (a & mask) | (b & ~mask);
}

u64 mutate(u64 genome, const Options& o, std::mt19937_64& rng, double rate) {
    std::bernoulli_distribution flip(rate);
    for (unsigned bit = 0; bit < 64; ++bit) {
        if (flip(rng)) genome ^= (u64{1} << bit);
    }
    // Occasional arithmetic mutation improves local exploration in numeric intervals.
    if (flip(rng)) {
        const u64 span = o.max_value - o.min_value;
        const u64 delta = random_value(rng, 0, std::max<u64>(1, span / 32));
        if (flip(rng)) genome = genome > delta ? genome - delta : o.min_value;
        else if (std::numeric_limits<u64>::max() - genome >= delta) genome += delta;
    }
    return clamp_genome(genome, o);
}

void print_individual(std::size_t generation, const Individual& best) {
    std::cout << "generation=" << generation
              << " candidate=" << best.genome
              << " stopping_time=" << best.fitness.steps
              << " peak=" << best.fitness.peak
              << " converged=" << (best.fitness.complete ? "true" : "false") << '\n';
}
} // namespace

int main(int argc, char** argv) {
    Options options;
    if (!parse(argc, argv, options)) {
        std::cerr << "Invalid arguments. Run scollatz-ga --help.\n";
        return 2;
    }

    const u64 seed = options.seed == 0
        ? static_cast<u64>(std::chrono::high_resolution_clock::now().time_since_epoch().count())
        : options.seed;
    std::mt19937_64 rng(seed);
    std::vector<Individual> population(options.population);
    for (auto& individual : population)
        individual.genome = random_value(rng, options.min_value, options.max_value);

    const auto started = std::chrono::steady_clock::now();
    double mutation_rate = options.mutation;
    Individual global_best{};
    bool have_best = false;

    for (std::size_t generation = 0; generation < options.generations; ++generation) {
        evaluate_population(population, options);
        std::sort(population.begin(), population.end(), better);
        if (!have_best || better(population.front(), global_best)) {
            global_best = population.front();
            have_best = true;
        }
        print_individual(generation, global_best);

        if (generation + 1 == options.generations) break;
        std::vector<Individual> next;
        next.reserve(options.population);
        for (std::size_t i = 0; i < options.elite; ++i) next.push_back(population[i]);

        // Adapt mutation upward after stagnation-like generations; keep it bounded.
        if (generation > 0 && population.front().fitness.steps == population[options.elite - 1].fitness.steps)
            mutation_rate = std::min(0.35, std::max(options.mutation, mutation_rate * 1.08));
        else
            mutation_rate = std::max(options.mutation * 0.5, mutation_rate * 0.96);

        while (next.size() < options.population) {
            const auto& p1 = tournament_pick(population, rng, options.tournament);
            const auto& p2 = tournament_pick(population, rng, options.tournament);
            u64 child = crossover(p1.genome, p2.genome, rng);
            child = mutate(child, options, rng, mutation_rate);
            next.push_back({child, {}});
        }
        population.swap(next);
    }

    const double elapsed = std::chrono::duration<double>(
        std::chrono::steady_clock::now() - started).count();
    std::cout << "best_candidate=" << global_best.genome
              << " stopping_time=" << global_best.fitness.steps
              << " peak=" << global_best.fitness.peak
              << " converged=" << (global_best.fitness.complete ? "true" : "false")
              << "\nseed=" << seed
              << " generations=" << options.generations
              << " population=" << options.population
              << " threads=" << options.threads
              << " elapsed_seconds=" << std::fixed << std::setprecision(3) << elapsed << '\n';
    return global_best.fitness.complete ? 0 : 1;
}
