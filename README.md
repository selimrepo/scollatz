# scollatz

C++20 Collatz scanner and island-model genetic search with multithreaded CPU and optional OpenCL fitness evaluation.

## Build
```sh
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
ctest --test-dir build --output-on-failure
```
Debian/Ubuntu OpenCL + PoCL: `sudo apt-get install cmake g++ ocl-icd-opencl-dev pocl-opencl-icd`. Install a vendor OpenCL ICD for discrete GPU support.

## Genetic search
```sh
./build/scollatz-ga --backend cpu --population 2048 --generations 500 --islands 8 --threads 16 --min 1 --max 1000000000000 --seed 42
./build/scollatz-ga --backend opencl --population 2048 --generations 500 --islands 8
./build/scollatz-ga --backend hybrid --population 2048 --generations 500 --islands 8 --threads 16
./build/scollatz-ga --backend auto --resume scollatz.chk --generations 1000
```
- `--backend auto|cpu|opencl|hybrid`: auto uses OpenCL with CPU fallback; hybrid evaluates half the population on OpenCL and half on CPU concurrently.
- `--islands N`: independent populations; every migration interval, top `--migration-count` individuals migrate in a ring.
- `--migration-interval N --migration-count N`: base island migration cadence and elite migrant count.
- Adaptive migration is enabled by default: low genome diversity increases migration frequency and migrant count; high diversity reduces migration pressure. Disable it with `--no-adaptive-migration`.
- Each island owns an independent OpenCL command queue, kernel instance, and persistent input/output buffers. Device allocations are reused across generations rather than recreated for each fitness batch.
- `--checkpoint FILE`: JSON checkpoint after every generation. `--resume FILE` restores generation, island genomes and PRNG states.
- `--csv FILE --json FILE`: per-generation, per-island best fitness records.
- Other controls: `--population`, `--generations`, `--elite`, `--tournament`, `--mutation`, `--threads`, `--min`, `--max`, `--max-steps`, `--seed`.

Fitness is exact stopping time with peak as tie-breaker, but the evolutionary search is heuristic and is not a proof of the Collatz conjecture. OpenCL accelerates candidate fitness; each island has its own queue and persistent buffers. Genetic operators and checkpoint I/O remain on CPU.

## Rust implementation

The Rust binaries are implemented under `rust/` and retain the C++ executables as a reference implementation.

- `scollatz`: multi-threaded CPU range scanning and OpenCL range scanning (`auto`, `cpu`, `opencl`).
- `scollatz-ga`: island-model GA with CPU, OpenCL and hybrid CPU/GPU fitness evaluation, migration and adaptive migration.
- Checkpoints use the C++ version-1 JSON format, including serialized `std::mt19937_64` state, so C++ and Rust can resume one another's checkpoints.
- The C++ and Rust GA implementations use the same explicitly defined MT19937-64, multiply-high integer sampling, and Bernoulli draw algorithm, independent of the C++ standard-library distribution implementation. CI compares full seeded checkpoints and both cross-language resume directions.

### Build and test

Install an OpenCL ICD/SDK and GPU driver (or PoCL for CPU-based OpenCL testing), then:

```sh
cargo test --manifest-path rust/Cargo.toml --all-targets
cargo build --release --manifest-path rust/Cargo.toml --bins
cargo run --release --manifest-path rust/Cargo.toml --bin scollatz -- --start 1 --end 1000000 --backend auto --threads 8
cargo run --release --manifest-path rust/Cargo.toml --bin scollatz-ga -- --backend hybrid --population 2048 --generations 500 --islands 8 --threads 16 --seed 42
```

Use `--backend cpu` to force CPU, `--backend opencl` to require OpenCL, and `--backend hybrid` to split each island's fitness batch between CPU and OpenCL. `auto` prefers OpenCL and falls back to CPU if no OpenCL device is available.


## Complex-analytic branch explorer

The Rust workspace also provides `scollatz-complex`, a research tool for studying the
complex affine extensions of the accelerated Collatz branches:

- even branch: `E(z) = z/2`
- odd branch: `O(z) = (3z + 1)/2`

A finite symbolic parity word (for example `10100`) determines a holomorphic affine
composition `F(z) = a*z + b`. The tool reports its complex orbit, multiplier, fixed
point (when defined), and local attracting/repelling classification. Supply either a
parity word directly or an integer seed whose accelerated trajectory reaches 1.

```sh
cargo run --release --manifest-path rust/Cargo.toml --bin scollatz-complex -- --word 10100 --real 2.5 --imag 1.25 --repeats 20
cargo run --release --manifest-path rust/Cargo.toml --bin scollatz-complex -- --seed 27 --repeats 10
cargo test --manifest-path rust/Cargo.toml --all-targets
```

**Mathematical limitation:** parity is defined for integers, not arbitrary complex
numbers. This program extends each integer branch holomorphically after fixing a
symbolic parity itinerary; it does not define a unique global complex Collatz map and
does not prove the Collatz conjecture. Its purpose is to investigate branch
compositions, multipliers, fixed points, and candidate periodic itineraries without
misrepresenting numerical evidence as a proof.


## Advanced mathematical research engine

The Rust workspace includes `scollatz-research`, which complements the floating-point
complex branch explorer with exact arithmetic:

- **Complex dynamics / affine dynamics:** each fixed symbolic itinerary is an affine
  holomorphic map on ℂ. Its multiplier is exactly `3^o / 2^k` and its fixed point
  is represented as a rational number embedded in ℂ. The modulus determines whether
  that prescribed branch composition is attracting, repelling, or neutral.
- **Analytic-number-theory structure:** every word is represented by the exact
  numerator map `(3^o*n + b)/2^k`; the number of odd branches controls its multiplier.
  The engine reports contraction/expansion counts by word length.
- **Rigorous finite cycle checks:** arbitrary-precision `BigInt` and `BigRational`
  arithmetic solve the fixed-point equation without floating-point rounding. A
  candidate is accepted only if it is a positive integer and exact accelerated
  iteration verifies the full parity itinerary and return to the starting value.

Run it with:

```sh
cargo test --manifest-path rust/Cargo.toml --all-targets
cargo run --release --manifest-path rust/Cargo.toml --bin scollatz-research -- --max-period 16
```

The exhaustive search is exponential and currently limits `--max-period` to 24.
It proves only the finite search statement for the configured word lengths. It cannot
exclude longer cycles, divergent trajectories, or establish the Collatz conjecture.
A global proof would require a mathematical argument covering all positive integers
and all possible orbit lengths; finite computation is evidence, not that argument.
