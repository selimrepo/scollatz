# scollatz

C++20 Collatz scanner and island-model genetic search with multithreaded CPU and optional OpenCL fitness evaluation.

## Build
```sh
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
ctest --test-dir build --output-on-failure
``
Debian/Ubuntu OpenCL + PoCL: `sudo apt-get install cmake g++ ocl-icd-opencl-dev pocl-opencl-icd`. Install a vendor OpenCL ICD for discrete GPU support.

## Genetic search
```sh
./build/scollatz-ga --backend cpu --population 2048 --generations 500 --islands 8 --threads 16 --min 1 --max 1000000000000 --seed 42
./build/scollatz-ga --backend opencl --population 2048 --generations 500 --islands 8
./build/scollatz-ga --backend hybrid --population 2048 --generations 500 --islands 8 --threads 16
./build/scollatz-ga --backend auto --resume scollatz.chk --generations 1000
``
- `--backend auto|cpu|opencl|hybrid`: auto uses OpenCL with CPU fallback; hybrid evaluates half the population on OpenCL and half on CPU concurrently.
- `--islands N`: independent populations; every migration interval, top `--migration-count` individuals migrate in a ring.
- `--migration-interval N --migration-count N`: island migration cadence and elite migrant count.
- `--checkpoint FILE`: JSON checkpoint after every generation. `--resume FILE` restores generation, island genomes and PRNG states.
- `--csv FILE --json FILE`: per-generation, per-island best fitness records.
- Other controls: `--population`, `--generations`, `--elite`, `--tournament`, `--mutation`, `--threads`, `--min`, `--max`, `--max-steps`, `--seed`.

Fitness is exact stopping time with peak as tie-breaker, but the evolutionary search is heuristic and is not a proof of the Collatz conjecture. OpenCL accelerates candidate fitness; genetic operators and checkpoint I/O remain on CPU.
