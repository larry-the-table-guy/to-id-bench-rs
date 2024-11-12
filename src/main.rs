#![cfg_attr(
    all(feature = "nightly", target_arch = "x86_64"),
    feature(avx512_target_feature)
)]
#![cfg_attr(
    all(feature = "nightly", target_arch = "x86_64"),
    feature(stdarch_x86_avx512)
)]

// Lowercase + retain [a-z0-9]
// These don't fully handle unicode - non-ascii bytes will be unconditionally removed (some unicode letters become ascii when lowercased)

fn to_id_match_16(input: &[u8; 16]) -> ([u8; 16], u8) {
    let mut bytes = [0u8; 16];
    let mut num_bytes = 0;
    for byte in input {
        let adjusted_byte = match *byte {
            b'a'..=b'z' | b'0'..=b'9' => *byte,
            b'A'..=b'Z' => byte + 32,
            _ => 0,
        };
        bytes[num_bytes] = adjusted_byte;
        num_bytes += (adjusted_byte != 0) as usize;
        // if byte != 0 { num_bytes += 1}; // equiv
    }
    (bytes, num_bytes as u8)
}

/// Maps the 256 bytes to 0 (skip) or a character (keep)
static ASCII_TABLE: &[u8; 256] = &{
    let mut t = [0u8; 256];
    let mut i = b'a';
    while i <= b'z' {
        t[i as usize] = i;
        i += 1;
    }
    i = b'A';
    while i <= b'Z' {
        t[i as usize] = i + 32;
        i += 1;
    }
    i = b'0';
    while i <= b'9' {
        t[i as usize] = i;
        i += 1;
    }
    t
};

fn to_id_table_16(input: &[u8; 16]) -> ([u8; 16], u8) {
    let mut bytes = [0u8; 16];
    let mut num_bytes = 0;
    for &byte in input {
        if byte >= 128 {
            continue;
        }
        let mapped = ASCII_TABLE[byte as usize];
        unsafe { *bytes.get_unchecked_mut(num_bytes) = mapped };
        num_bytes += (mapped != 0) as usize;
    }
    (bytes, num_bytes as u8)
}

fn to_id_full_table_16(input: &[u8; 16]) -> ([u8; 16], u8) {
    let mut bytes = [0u8; 16];
    let mut num_bytes = 0;
    for &byte in input {
        let mapped = ASCII_TABLE[byte as usize];
        unsafe { *bytes.get_unchecked_mut(num_bytes) = mapped };
        num_bytes += (mapped != 0) as usize;
    }
    (bytes, num_bytes as u8)
}

#[cfg(target_arch = "x86_64")]
mod x86_64 {
    use std::arch::x86_64::*;

    /// Returns (lowered & filtered bytes, active mask)
    #[inline(always)]
    unsafe fn lower_and_mask_helper_sse41(input: __m128i) -> (__m128i, __m128i) {
        use std::arch::x86_64::{
            _mm_and_si128 as and, _mm_cmpgt_epi8 as cmpgt, _mm_or_si128 as or,
            _mm_set1_epi8 as set1,
        };
        // ideally you'd run this in batches and these constants would get loaded outside the loop
        // exclusive bc sse doesn't have le, ge
        let lo_a_v = set1(b'a' as i8 - 1);
        let lo_z_v = set1(b'z' as i8 + 1);
        let up_a_v = set1(b'A' as i8 - 1);
        let up_z_v = set1(b'Z' as i8 + 1);
        let di_0_v = set1(b'0' as i8 - 1);
        let di_9_v = set1(b'9' as i8 + 1);
        let ascii_alpha_lower_bit_v = set1((b'a' - b'A') as i8); // 32

        let chunk = input;
        // find lower, upper, numbers
        let lo_mask = and(cmpgt(chunk, lo_a_v), cmpgt(lo_z_v, chunk));
        let up_mask = and(cmpgt(chunk, up_a_v), cmpgt(up_z_v, chunk));
        let di_mask = and(cmpgt(chunk, di_0_v), cmpgt(di_9_v, chunk));
        // lower the upper-case bytes
        let lowered = _mm_or_si128(chunk, ascii_alpha_lower_bit_v);
        // there's no masked or for epi8, so blend
        let chunk = _mm_blendv_epi8(chunk, lowered, up_mask);
        // keep bytes from those masks. Can do 2 ors in one step w/ vpternlogd
        let retain_mask = or(or(lo_mask, up_mask), di_mask);
        (chunk, retain_mask)
    }

    pub fn can_run_pext_16() -> bool {
        is_x86_feature_detected!("popcnt")
            && is_x86_feature_detected!("bmi2")
            && is_x86_feature_detected!("sse4.1")
    }

    #[target_feature(enable = "popcnt,bmi2,sse4.1")]
    pub unsafe fn to_id_pext_16(input: &[u8; 16]) -> ([u8; 16], u8) {
        // build mask for pext,
        // do pexts and concat (by unaligned writes)
        let chunk = _mm_loadu_si128(input.as_ptr().cast());
        let (chunk, retain_mask) = lower_and_mask_helper_sse41(chunk);
        let low_bytes = _mm_extract_epi64::<0>(chunk) as u64;
        let low_mask = _mm_extract_epi64::<0>(retain_mask) as u64;
        let high_bytes = _mm_extract_epi64::<1>(chunk) as u64;
        let high_mask = _mm_extract_epi64::<1>(retain_mask) as u64;
        let mut packed_bytes = [0u8; 16];
        let ptr = packed_bytes.as_mut_ptr().cast::<[u8; 8]>();
        ptr.write(_pext_u64(low_bytes, low_mask).to_le_bytes());
        ptr.byte_add(low_mask.count_ones() as usize / 8)
            .write(_pext_u64(high_bytes, high_mask).to_le_bytes());
        (
            packed_bytes,
            _mm_movemask_epi8(retain_mask).count_ones() as u8,
        )
    }

    #[cfg(feature = "nightly")]
    pub fn can_run_avx512_16() -> bool {
        is_x86_feature_detected!("popcnt")
            && is_x86_feature_detected!("avx512f")
            && is_x86_feature_detected!("avx512bw")
            && is_x86_feature_detected!("avx512vbmi2")
            && is_x86_feature_detected!("avx512vl")
    }

    #[cfg(feature = "nightly")]
    #[target_feature(enable = "popcnt,avx512f,avx512bw,avx512vbmi2,avx512vl")]
    pub unsafe fn to_id_avx512_16(input: &[u8; 16]) -> ([u8; 16], u8) {
        let (chunk, retain_mask) =
            lower_and_mask_helper_sse41(_mm_loadu_si128(input.as_ptr().cast()));
        let retain_mask = _mm_movemask_epi8(retain_mask) as u16;
        let num_bytes = retain_mask.count_ones() as u8;
        let packed_bytes = std::mem::transmute(_mm_maskz_compress_epi8(retain_mask, chunk));
        (packed_bytes, num_bytes)
    }

    #[cfg(feature = "nightly")]
    pub fn can_run_avx512_lut_16() -> bool {
        is_x86_feature_detected!("popcnt")
            && is_x86_feature_detected!("avx512f")
            && is_x86_feature_detected!("avx512bw")
            && is_x86_feature_detected!("avx512vbmi")
            && is_x86_feature_detected!("avx512vbmi2")
            && is_x86_feature_detected!("avx512vl")
    }

    /// use LUT + compress
    #[cfg(feature = "nightly")]
    #[target_feature(enable = "popcnt,avx512f,avx512vl,avx512vbmi,avx512vbmi2,avx512bw")]
    pub unsafe fn to_id_avx512_lut_16(input: &[u8; 16]) -> ([u8; 16], u8) {
        use crate::ASCII_TABLE;
        let low_lut = _mm512_loadu_si512(ASCII_TABLE.as_ptr().cast());
        let high_lut = _mm512_loadu_si512(ASCII_TABLE.as_ptr().byte_add(64).cast());
        // use a 128 byte LUT to map bytes. Nonzero bytes are retained.

        let b = _mm_loadu_si128(input.as_ptr().cast());
        // which lanes don't have 8th bit set
        let ascii_mask = !_mm_test_epi8_mask(b, _mm_set1_epi8(-128));
        // this is a no-op, just for typing
        let b = _mm512_zextsi128_si512(b);
        let mapped_bytes = _mm512_permutex2var_epi8(low_lut, b, high_lut);
        // cast back down to the 16 actual bytes
        let mapped_bytes = _mm512_castsi512_si128(mapped_bytes);
        // ignore bytes >= 128 (not ascii), get mask of nonzero bytes
        let retain_mask = _mm_mask_test_epi8_mask(ascii_mask, mapped_bytes, mapped_bytes) as u16;
        let packed_bytes = _mm_maskz_compress_epi8(retain_mask, mapped_bytes);
        (
            std::mem::transmute(packed_bytes),
            retain_mask.count_ones() as u8,
        )
    }

    #[cfg(feature = "nightly")]
    pub fn can_run_avx512_lut_4x16() -> bool {
        is_x86_feature_detected!("popcnt")
            && is_x86_feature_detected!("avx512f")
            && is_x86_feature_detected!("avx512bitalg")
            && is_x86_feature_detected!("avx512bw")
            && is_x86_feature_detected!("avx512vbmi")
            && is_x86_feature_detected!("avx512vbmi2")
            && is_x86_feature_detected!("avx512vl")
            && is_x86_feature_detected!("avx512vpopcntdq")
    }

    /// LUT but 4 at a time. Returns a dummy u8 because I am a dummy.
    #[cfg(feature = "nightly")]
    #[target_feature(enable = "avx512f,avx512vl,avx512vbmi,avx512vbmi2,avx512bw,avx512vpopcntdq,avx512bitalg")]
    pub unsafe fn to_id_avx512_lut_4x16(inputs: &[u8; 64]) -> ([u8; 64], u8) {
        use crate::ASCII_TABLE;
        let low_lut = _mm512_loadu_si512(ASCII_TABLE.as_ptr().cast());
        let high_lut = _mm512_loadu_si512(ASCII_TABLE.as_ptr().byte_add(64).cast());
        // use a 128 byte LUT to map bytes. Nonzero bytes are retained.

        let b = _mm512_loadu_si512(inputs.as_ptr().cast());
        // which lanes don't have 8th bit set
        let ascii_mask = !_mm512_test_epi8_mask(b, _mm512_set1_epi8(-128));
        let mapped_bytes = _mm512_permutex2var_epi8(low_lut, b, high_lut);
        // ignore bytes >= 128 (not ascii), get mask of nonzero bytes
        let retain_mask = _mm512_mask_test_epi8_mask(ascii_mask, mapped_bytes, mapped_bytes) as u64;
        // now, this compress clumps all 4 strings together. We'll need to expand them back into
        // their rightful places
        let packed_bytes = _mm512_maskz_compress_epi8(retain_mask, mapped_bytes);
        // we need a bitmask for 4x16 - can compute from popcount of each 16-bit quadrant of retain
        let retain_v = _mm_set_epi64x(0, retain_mask as i64);
        let retain_cnt_v = _mm_popcnt_epi16(retain_v);
        // now turn counts into bitmasks (5 -> 0b1_1111)
        let masks_v = _mm_sub_epi16(_mm_sllv_epi16(_mm_set1_epi16(1), retain_cnt_v), _mm_set1_epi16(1));
        let expand_mask = _mm_cvtsi128_si64(masks_v) as u64;
        let slotted_bytes = _mm512_maskz_expand_epi8(expand_mask, packed_bytes);
        (std::mem::transmute(slotted_bytes), 0)
    }
}
type CompatTestFn = fn() -> bool;

/// Stages of the benchmark. Order dependent
static BENCH_MODES: &[(&'static str, fn(&mut Vec<[u8; 16]>))] = &[
    ("Random binary", |_| {}),
    ("Random ascii", |vec| {
        for chunk in vec.iter_mut() {
            for byte in chunk.iter_mut() {
                *byte &= 0x7F;
            }
        }
    }),
    ("Random alpha-num", |vec| {
        for chunk in vec.iter_mut() {
            for byte in chunk.iter_mut() {
                let k = *byte % (26 + 26 + 10);
                *byte = match k {
                    0..=25 => k + b'a',
                    26..=51 => k + b'A',
                    _ => k + b'0',
                };
            }
        }
    }),
];

fn fxhash(h: u64, w: u64) -> u64 {
    (h.rotate_left(5) ^ w).wrapping_mul(0x517cc1b727220a95)
}

fn main() {
    use std::{hint::black_box, time::Instant};

    #[inline(never)]
    fn bench<const CHUNK_SIZE: usize>(
        fn_label: &str,
        compat_test: CompatTestFn,
        to_id_fn: impl Fn(&[u8; CHUNK_SIZE]) -> ([u8; CHUNK_SIZE], u8),
        inputs: &[[u8; CHUNK_SIZE]],
    ) {
        let mut out_buf = [std::mem::MaybeUninit::<[u8; CHUNK_SIZE]>::uninit(); 1000];
        if !compat_test() {
            println!("--Skipping '{fn_label}' because of missing CPU features");
            return;
        }
        let start = Instant::now();
        for chunks in inputs.chunks_exact(out_buf.len() / (CHUNK_SIZE / 16)) {
            for (in_chunk, out_chunk) in chunks.iter().zip(out_buf.iter_mut()) {
                out_chunk.write(to_id_fn(in_chunk).0);
            }
            black_box(out_buf);
        }
        let duration = start.elapsed().as_secs_f64();
        let thrpt = inputs.len() as f64 / duration / 1_000_000. * (CHUNK_SIZE / 16) as f64;
        println!("{fn_label:<18}: {thrpt:>7.2} M/s");
    }

    let num_iters = 10_000_000u32;
    // very crappy means of generating random input
    let mut inputs = Vec::<[u8; 16]>::with_capacity(num_iters as usize);
    let mut h = 0;
    for i in 0..num_iters {
        h = fxhash(h, i as u64);
        let w1 = h;
        h = fxhash(h, i as u64);
        let w2 = h;
        let mut chunk = [0u8; 16];
        chunk[..8].copy_from_slice(&w1.to_le_bytes());
        chunk[8..].copy_from_slice(&w2.to_le_bytes());
        inputs.push(chunk);
    }

    for (mode_label, input_prep_fn) in BENCH_MODES {
        println!("\n\tStarting '{mode_label}'");
        input_prep_fn(&mut inputs);
        bench("scalar match", || true, to_id_match_16, &inputs);
        bench("scalar table-128", || true, to_id_table_16, &inputs);
        bench("scalar table-256", || true, to_id_full_table_16, &inputs);
        #[cfg(target_arch = "x86_64")]
        {
            use x86_64::*;
            bench(
                "pext",
                can_run_pext_16,
                |v| unsafe { to_id_pext_16(v) },
                &inputs,
            );
            #[cfg(feature = "nightly")]
            {
                bench(
                    "AVX512 Blend",
                    can_run_avx512_16,
                    |v| unsafe { to_id_avx512_16(v) },
                    &inputs,
                );
                bench(
                    "AVX512 LUT",
                    can_run_avx512_lut_16,
                    |v| unsafe { to_id_avx512_lut_16(v) },
                    &inputs,
                );
                // turn slice of 16 into 64
                let inputs_64 = unsafe { std::slice::from_raw_parts(inputs.as_ptr().cast::<[u8; 64]>(), inputs.len() / 4) };
                bench(
                    "AVX512 LUT x 4",
                    can_run_avx512_lut_4x16,
                    |v| unsafe { to_id_avx512_lut_4x16(v) },
                    inputs_64,
                );
            }
        }
    }
}

#[cfg(test)]
mod test {
    use crate::*;

    static TEST_CASES: &[([u8; 16], [u8; 16])] = &[
        (
            *b"\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0",
            *b"\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0",
        ),
        (
            *b"A\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0",
            *b"a\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0",
        ),
        (
            *b"\0a\0\0\0\0\0\0\0\0\0\0\0\0\0\0",
            *b"a\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0",
        ),
        (
            *b"/a/[@\0\0\0\0\0\0\0\0\0\0\0",
            *b"a\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0",
        ),
        (
            *b"\02\0\0\0\0\0\0\0\0\0\0\0\0\0\0",
            *b"2\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0",
        ),
        (
            *b"+2@3^#-0\0\0\0\0\0\0\09",
            *b"2309\0\0\0\0\0\0\0\0\0\0\0\0",
        ),
        (
            *b"\0$\0\0\0\0\0\0\0\0\0\0\0\0\0\0",
            *b"\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0",
        ),
        (
            *b"\0a\0b\08\0\0\0\0\0\0\0\0\0\0",
            *b"ab8\0\0\0\0\0\0\0\0\0\0\0\0\0",
        ),
        (
            *b"\0a\0b\0-2\0\0\0\0\0\0\0\0\0",
            *b"ab2\0\0\0\0\0\0\0\0\0\0\0\0\0",
        ),
        (*b"A-@s az0+!_+pDkV", *b"asaz0pdkv\0\0\0\0\0\0\0"),
        (
            [
                128, 129, 130, 140, 150, 160, 170, 180, b'a', 200, 250, 251, 252, 253, 254, 255,
            ],
            *b"a\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0",
        ),
    ];

    #[test]
    fn test_scalar_match() {
        for (i, o) in TEST_CASES {
            let (b, _l) = to_id_match_16(i);
            assert_eq!(*o, b);
        }
    }

    #[test]
    fn test_scalar_table() {
        for (i, o) in TEST_CASES {
            let (b, _l) = to_id_table_16(i);
            assert_eq!(*o, b);
        }
    }

    #[test]
    fn test_scalar_full_table() {
        for (i, o) in TEST_CASES {
            let (b, _l) = to_id_full_table_16(i);
            assert_eq!(*o, b);
        }
    }

    #[cfg(target_arch = "x86_64")]
    #[test]
    fn test_pext() {
        if !x86_64::can_run_pext_16() {
            return;
        }
        for (i, o) in TEST_CASES {
            let (b, _l) = unsafe { x86_64::to_id_pext_16(i) };
            assert_eq!(*o, b);
        }
    }

    #[cfg(all(feature = "nightly", target_arch = "x86_64"))]
    // naming is hard
    #[test]
    fn test_avx512_first() {
        if !x86_64::can_run_avx512_16() {
            return;
        }
        for (i, o) in TEST_CASES {
            let (b, _l) = unsafe { x86_64::to_id_avx512_16(i) };
            assert_eq!(*o, b);
        }
    }

    #[cfg(all(feature = "nightly", target_arch = "x86_64"))]
    #[test]
    fn test_avx512_lut() {
        if !x86_64::can_run_avx512_lut_16() {
            return;
        }
        for (i, o) in TEST_CASES {
            let (b, _l) = unsafe { x86_64::to_id_avx512_lut_16(i) };
            assert_eq!(*o, b);
        }
    }

    #[cfg(all(feature = "nightly", target_arch = "x86_64"))]
    #[test]
    fn test_avx512_lut_x4() {
        if !x86_64::can_run_avx512_lut_4x16() {
            return;
        }
        for cs in TEST_CASES.chunks_exact(4) {
            let mut i_buf = [0u8; 64];
            i_buf[..16].copy_from_slice(&cs[0].0);
            i_buf[16..32].copy_from_slice(&cs[1].0);
            i_buf[32..48].copy_from_slice(&cs[2].0);
            i_buf[48..].copy_from_slice(&cs[3].0);

            let mut o_buf = [0u8; 64];
            o_buf[..16].copy_from_slice(&cs[0].1);
            o_buf[16..32].copy_from_slice(&cs[1].1);
            o_buf[32..48].copy_from_slice(&cs[2].1);
            o_buf[48..].copy_from_slice(&cs[3].1);
            let (bs, _l) = unsafe { x86_64::to_id_avx512_lut_4x16(&i_buf) };
            assert_eq!(o_buf, bs);
        }
    }
}
