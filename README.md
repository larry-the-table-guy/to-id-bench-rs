A small benchmark comparing several approaches to `toID` on fixed-length strings.

Why fixed-length strings? Well, I've found that when you want a system that deals with many small strings to be fast,
it can really help to customize your string representation. Having fixed-length inline strings removes a level of
indirection and potentially many branches. In 64bit Rust programs, string slices (`&str`) take 16 bytes anyway.
Of course, this approach would require a separate path for large strings; depending on the problem, that may turn out to not be very much code.

In practice, you'd want this function inlined in the hot loop, and possibly running on batches of strings.
That's the scenario the benchmark tests.

For the sake of having no dependencies and keeping it within one file, I've compromised on the testing and benchmark code.

# How to use
Requires Rust 1.89.0 or later.

LLVM won't (or is much less likely to) inline the unsafe fns unless you build with the features enabled.
Easiest thing is to use the 'native' target. However, sometimes LLVM's metadata is outdated and it won't recognize
your CPU (it will underestimate its capabilities).

```sh
# not all optimizations will be present
cargo run --release

# targets host CPU, non-portable
RUSTFLAGS="-Ctarget-cpu=native" cargo run --release
```

or

```sh
rustc -Copt-level=3 -Ctarget-cpu=native ./src/main.rs
./main
```

# My Results
W/ `target-cpu=native`. Throughput in `million strings / second`.

Obviously, in practice you won't have such large batches, but this is still
demonstrative of which algos are faster, and what the potential is.

## 9th Gen Intel

| Fn | Binary | Ascii | Alphanum |
| -- | ------ | ----- | -------- |
| scalar match | 48 | 48 | 48 |
| scalar table-128 | 15 | 60 | 60 |
| scalar table-256 | 111 | 109 | 111 |
| pext | 167 | 167 | 165 |
| AVX512 Blend | - | - | - |
| AVX512 LUT | - | - | - |

## CPU w/ AVX512

| Fn | Binary | Ascii | Alphanum |
| -- | ------ | ----- | -------- |
| scalar match | 69 | 80 | 80 |
| scalar table-128 | 22 | 123 | 124 |
| scalar table-256 | 147 | 169 | 170 |
| pext | 212 | 226 | 222 |
| AVX512 Blend | 1010 | 1015 | 1016 |
| AVX512 LUT | 1396 | 1379 | 1347 |
