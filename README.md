A small benchmark comparing several approaches to `toID` on fixed-length strings.

Why fixed-length strings? Well, I've found that when you want a system that deals with many small strings to be fast,
it can really help to customize your string representation. Having fixed-length inline strings removes a level of
indirection and potentially many branches. In 64bit Rust programs, string slices (`&str`) take 16 bytes anyway.
Of course, this approach would require a separate path for large strings; depending on the problem, that may not turn out to be very much code.

In practice, you'd want this function inlined in the hot loop, and possibly running on batches of strings.
In the current state, constants are loaded on every call. That increases the function size and adds several cycles of latency.
This negatively affects the `pext` and `avx512` functions the most.

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
