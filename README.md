A small benchmark comparing several approaches to `toID` on fixed-length strings.

Why fixed-length strings? Well, I've found that when you want a system that deals with many small strings to be fast,
it can really help to customize your string representation. Having fixed-length inline strings removes a level of
indirection and potentially many branches. In 64bit Rust programs, string slices (`&str`) take 16 bytes anyway.
Of course, this approach would require a separate path for large strings; depending on the problem, that may not turn out to be very much code.

In practice, you'd want this function inlined in the hot loop, and possibly running on batches of strings.
That's the scenario the benchmark tests.

For the sake of having no dependencies and keeping it within one file, I've compromised on the testing and benchmark code.

# How to use
LLVM won't (or is much less likely to) inline the unsafe fns unless you build with the features enabled.
Easiest thing is to use the 'native' target. However, sometimes LLVM's metadata is outdated and it won't recognize
your CPU (it will underestimate its capabilities).

```
RUSTFLAGS="-Ctarget-cpu=native" cargo run --release
```

or

```
rustc -Copt-level=3 -Ctarget-cpu=native ./src/main.rs
./main
```

# My Results
W/ `target-cpu=native`. Time in `ns`. Precision isn't great.

## 9th Gen Intel

| Fn | Binary | Ascii | Alphanum |
| -- | ------ | ----- | -------- |
| scalar match | 24 | 24 | 26 |
| scalar table-128 | 52 | 12 | 12 |
| scalar table-256 | 8 | 8 | 8 |
| pext | 5 | 5 | 4 |
| AVX512 Blend | - | - | - |
| AVX512 LUT | - | - | - |

## CPU w/ AVX512

| Fn | Binary | Ascii | Alphanum |
| -- | ------ | ----- | -------- |
| scalar match | 14 | 12 | 12 |
| scalar table-128 | 42 | 6 | 6 |
| scalar table-256 | 6 | 5 | 5 |
| pext | 4 | 4 | 2 |
| AVX512 Blend | 0 | 0 | 0 |
| AVX512 LUT | 0 | 0 | 0 |

Obviously, it's pretty unreasonable for the AVX512 variants to be under 1 ns.
I suspect that the CPU is recognizing that each write will dominate the last write,
and some spec exec / instruction reordering shenanigans are afoot.
On the other hand, with 5+ GHz, 4-6 IPC *and* a large reorder buffer, it could just be the CPU being really fast and pipelining across function calls...

I'll look at changing the benchmark to write the outputs into a small buffer.
