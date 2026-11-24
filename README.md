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

## Rust port (in progress)

A Rust implementation is being developed under `rust/` alongside the C++ reference implementation. It currently provides the CPU range scanner and a CPU island-model GA with migration, adaptive migration, checkpointing, and CSV/JSON generation.

Build and test it with:
```sh
cargo test --manifest-path rust/Cargo.toml --all-targets
cargo build --release --manifest-path rust/Cargo.toml --bins
cargo run --release --manifest-path rust/Cargo.toml --bin scollatz -- --start 1 --end 1000000 --backend cpu --threads 8
cargo run --release --manifest-path rust/Cargo.toml --bin scollatz-ga -- --backend cpu --population 2048 --generations 500 --islands 8 --threads 16 --seed 42
```

**Parity status:** this is an additive port, not yet a drop-in replacement. The C++ implementation remains the feature-complete reference. Rust currently implements CPU execution; Rust OpenCL/hybrid execution, C++ checkpoint compatibility (including serialized MT19937-64 state), and exact cross-language seeded GA replay are not yet implemented. Do not remove the C++ targets until those parity items have dedicated tests.
