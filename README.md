# scollatz

A high-throughput Collatz stopping-time and peak-value scanner with a multithreaded C++20 CPU backend and an optional OpenCL GPU backend.

## Features

- Parallel CPU execution with dynamically claimed chunks and configurable worker count.
- OpenCL GPU kernel: one work-item per starting integer, chunked host/device transfers.
- Automatic GPU selection with CPU fallback in `auto` mode.
- Explicit 64-bit overflow detection; no silent arithmetic wraparound.
- Configurable iteration cap, inclusive scan range, and chunk size.
- Reports completed, overflowed and capped trajectories, maximum stopping time, maximum excursion and throughput.
- OpenCL is optional at build time; the CPU implementation has no third-party runtime dependency.

## Build

```sh
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
ctest --test-dir build --output-on-failure
```

To force a CPU-only build:

```sh
cmake -S . -B build -DSCOLLATZ_ENABLE_OPENCL=OFF -DCMAKE_BUILD_TYPE=Release
```

On Debian/Ubuntu, install OpenCL development files and an ICD implementation to enable GPU/CPU OpenCL devices:

```sh
sudo apt-get install ocl-icd-opencl-dev pocl-opencl-icd
```

For a discrete GPU, install the vendor's OpenCL runtime/ICD as well.

## Usage

```sh
./build/scollatz --start 1 --end 100000000 --backend auto
./build/scollatz --start 1 --end 10000000 --backend cpu --threads 16 --chunk 8192
./build/scollatz --start 1 --end 10000000 --backend opencl --chunk 262144
```

The range is inclusive. `auto` prefers an available OpenCL GPU and falls back to CPU. `--backend opencl` fails explicitly if no usable OpenCL GPU exists. A non-zero exit status indicates invalid input, runtime failure, overflow, or at least one trajectory reaching the iteration cap.

## Correctness and scope

This program scans independent starting values; it does not claim a proof of the Collatz conjecture. All trajectories use unsigned 64-bit arithmetic with checked `3n+1`. The default iteration cap prevents pathological or unexpectedly long runs from consuming unbounded time. Results are exact for trajectories that complete before overflow or the configured cap.

## CLI

| Option | Default | Description |
|---|---:|---|
| `--start N` | 1 | First starting value |
| `--end N` | 1,000,000 | Last starting value (inclusive) |
| `--backend auto|cpu|opencl` | auto | Execution backend |
| `--threads N` | hardware concurrency | CPU worker count |
| `--chunk N` | 4096 | Work allocation / GPU batch size |
| `--max-steps N` | 10,000,000 | Maximum iterations per trajectory |


## Genetic algorithm search

`scollatz-ga` is a heuristic evolutionary search for starting values with high Collatz stopping times. It uses a population of 64-bit genomes, tournament selection, uniform bitwise crossover, elitism, adaptive bit-flip mutation and occasional arithmetic mutations. Fitness is evaluated in parallel across CPU threads.

```sh
./build/scollatz-ga --population 1024 --generations 500 --min 1 --max 1000000000000 --threads 16 --mutation 0.04 --elite 32 --tournament 5 --seed 42
```

The GA is a search heuristic, not an exhaustive proof or a replacement for the exact scanner. Its result depends on the seed, search interval, population and generation budget. The reported fitness is the exact stopping time and peak for each candidate that completes within the configured iteration cap.
