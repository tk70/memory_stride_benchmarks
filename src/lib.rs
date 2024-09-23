#![feature(portable_simd)]
#![feature(test)]
#[deny(soft_unstable)]
extern crate test;

pub mod bench_pointer_arithmetic;
pub mod bench_runtime;
pub mod utils;

#[cfg(test)]
mod tests {
    use std::{
        hint::black_box,
        ops::Mul,
        simd::{Simd, StdFloat},
    };
    use test::Bencher;

    use crate::utils::*;

    /// data size
    const N_SMALL: usize = 9;

    /// data size
    const N_BIG: usize = 256;

    #[inline(always)]
    fn bench<T>(
        b: &mut Bencher,
        data_sets: &[T],
        result: &mut [Cluster],
        f: impl Fn(&T, &mut [Cluster]),
    ) {
        let mut i = 0;
        b.iter(|| {
            let data_set = &data_sets[i];
            black_box(f(data_set, result));
            i += 7;
            if i >= M {
                i -= M;
            }
        });
    }

    // ------------------------------------------------------------------------

    struct S3 {
        a: Cluster,
        b: Cluster,
        c: Cluster,
    }

    fn make_aos3(seed: usize, n: usize) -> Vec<S3> {
        let mut v = Vec::with_capacity(n);
        for i in seed..seed + n {
            v.push(S3 {
                a: Cluster::splat(i as f32),
                b: Cluster::splat((i / 3) as f32),
                c: Cluster::splat((i * 2) as f32),
            });
        }
        v
    }

    fn compute_aos3(data_set: &Vec<S3>, result: &mut [Cluster]) -> Cluster {
        let mut sum = Cluster::splat(0.0);
        for (s3, r) in data_set.iter().zip(result.iter_mut()) {
            let tmp = s3.a.mul_add(s3.b, s3.c);
            *r = tmp;
            sum += tmp;
        }
        sum
    }

    fn bench_aos3_impl(b: &mut Bencher, n: usize) {
        let data_sets = make_datasets(|i| make_aos3(i, n));
        let mut result = vec![Cluster::splat(0.0); n];
        bench(b, &data_sets, &mut result, |data_set, result| {
            black_box(compute_aos3(data_set, result));
        });
    }

    #[bench]
    fn bench_3_small_aos(b: &mut Bencher) {
        bench_aos3_impl(b, N_SMALL);
    }

    #[bench]
    fn bench_3_big_aos(b: &mut Bencher) {
        bench_aos3_impl(b, N_BIG);
    }

    // ------------------------------------------------------------------------

    struct SoA3 {
        a: Vec<Cluster>,
        b: Vec<Cluster>,
        c: Vec<Cluster>,
    }

    fn make_soa3(seed: usize, n: usize) -> SoA3 {
        let mut b = Vec::with_capacity(n);
        let mut c = Vec::with_capacity(n);
        let mut a = Vec::with_capacity(n);
        for i in seed..seed + n {
            a.push(Cluster::splat(i as f32));
            b.push(Cluster::splat((i / 3) as f32));
            c.push(Cluster::splat((i * 2) as f32));
        }
        SoA3 { a, b, c }
    }

    fn bench_soa3_impl(b: &mut Bencher, n: usize) {
        let data_sets = make_datasets(|i| make_soa3(i, n));
        let mut result = vec![Cluster::splat(0.0); n];
        bench(b, &data_sets, &mut result, |data_set, result| {
            black_box(compute_soa3(data_set, result));
        });
    }

    fn compute_soa3(data_set: &SoA3, result: &mut [Cluster]) -> Cluster {
        let mut sum = Cluster::splat(0.0);
        let n = data_set.a.len();
        for i in 0..n {
            let tmp = data_set.a[i].mul_add(data_set.b[i], data_set.c[i]);
            result[i] = tmp;
            sum += tmp;
        }
        sum
    }

    #[bench]
    fn bench_3_small_soa(b: &mut Bencher) {
        bench_soa3_impl(b, N_SMALL);
    }

    #[bench]
    fn bench_3_big_soa(b: &mut Bencher) {
        bench_soa3_impl(b, N_BIG);
    }

    #[test]
    fn test_3_benchmarks() {
        let aos3 = make_aos3(0, 10);
        let soa3 = make_soa3(0, 10);
        let mut result = vec![Cluster::splat(0.0); 10];
        assert_eq!(
            compute_aos3(&aos3, &mut result),
            compute_soa3(&soa3, &mut result)
        );
    }

    // ------------------------------------------------------------------------
    struct S5 {
        a: Cluster,
        b: Cluster,
        c: Cluster,
        d: Cluster,
        e: Cluster,
    }

    fn make_aos5(seed: usize, n: usize) -> Vec<S5> {
        let mut v = Vec::with_capacity(n);
        for i in seed..seed + n {
            v.push(S5 {
                a: Cluster::splat(i as f32),
                b: Cluster::splat((i / 3) as f32),
                c: Cluster::splat((i * 2) as f32),
                d: Cluster::splat((i + 3) as f32),
                e: Cluster::splat((i * 4) as f32),
            });
        }
        v
    }

    fn compute_aos5(data_set: &Vec<S5>, result: &mut [Cluster]) -> Cluster {
        let mut sum = Cluster::splat(0.0);
        for (s5, r) in data_set.iter().zip(result.iter_mut()) {
            let tmp = s5.a.mul_add(s5.b, s5.c).mul_add(s5.d, s5.e);
            *r = tmp;
            sum += tmp;
        }
        sum
    }

    fn bench_aos5_impl(b: &mut Bencher, n: usize) {
        let data_sets = make_datasets(|i| make_aos5(i, n));
        let mut result = vec![Cluster::splat(0.0); n];
        bench(b, &data_sets, &mut result, |data_set, result| {
            black_box(compute_aos5(data_set, result));
        });
    }

    #[bench]
    fn bench_5small_aos(b: &mut Bencher) {
        bench_aos5_impl(b, N_SMALL);
    }

    #[bench]
    fn bench_5big_aos(b: &mut Bencher) {
        bench_aos5_impl(b, N_BIG);
    }

    // ------------------------------------------------------------------------

    struct SoA5 {
        a: Vec<Cluster>,
        b: Vec<Cluster>,
        c: Vec<Cluster>,
        d: Vec<Cluster>,
        e: Vec<Cluster>,
    }

    fn make_soa5(seed: usize, n: usize) -> SoA5 {
        let mut b = Vec::with_capacity(n);
        let mut a = Vec::with_capacity(n);
        let mut c = Vec::with_capacity(n);
        let mut e = Vec::with_capacity(n);
        let mut d = Vec::with_capacity(n);
        for i in seed..seed + n {
            a.push(Cluster::splat(i as f32));
            b.push(Cluster::splat((i / 3) as f32));
            c.push(Cluster::splat((i * 2) as f32));
            d.push(Cluster::splat((i + 3) as f32));
            e.push(Cluster::splat((i * 4) as f32));
        }
        SoA5 { a, b, c, d, e }
    }

    fn bench_soa5_impl(b: &mut Bencher, n: usize) {
        let data_sets = make_datasets(|i| make_soa5(i, n));
        let mut result = vec![Cluster::splat(0.0); n];
        bench(b, &data_sets, &mut result, |data_set, result| {
            black_box(compute_soa5(data_set, result));
        });
    }

    fn compute_soa5(data_set: &SoA5, result: &mut [Cluster]) -> Cluster {
        let mut sum = Cluster::splat(0.0);
        let n = data_set.a.len();
        for i in 0..n {
            let tmp = data_set.a[i]
                .mul_add(data_set.b[i], data_set.c[i])
                .mul_add(data_set.d[i], data_set.e[i]);
            result[i] = tmp;
            sum += tmp;
        }
        sum
    }

    #[bench]
    fn bench_5small_soa(b: &mut Bencher) {
        bench_soa5_impl(b, N_SMALL);
    }

    #[bench]
    fn bench_5big_soa(b: &mut Bencher) {
        bench_soa5_impl(b, N_BIG);
    }

    // ------------------------------------------------------------------------

    struct AoSblob5 {
        data: Vec<Cluster>,
    }

    fn make_aos5_blob(seed: usize, n: usize) -> AoSblob5 {
        let mut data = Vec::with_capacity(n * 5);
        for i in seed..seed + n {
            data.push(Cluster::splat(i as f32));
            data.push(Cluster::splat((i / 3) as f32));
            data.push(Cluster::splat((i * 2) as f32));
            data.push(Cluster::splat((i + 3) as f32));
            data.push(Cluster::splat((i * 4) as f32));
        }
        AoSblob5 { data }
    }

    fn compute_aos_blob5(data_set: &AoSblob5, result: &mut [Cluster]) -> Cluster {
        let mut sum = Cluster::splat(0.0);
        let mut i = 0;
        for r in result.iter_mut() {
            let tmp = data_set.data[i]
                .mul_add(data_set.data[i + 1], data_set.data[i + 2])
                .mul_add(data_set.data[i + 3], data_set.data[i + 4]);
            *r = tmp;
            sum += tmp;
            i += 5;
        }
        sum
    }

    fn bench_aos_blob5_impl(b: &mut Bencher, n: usize) {
        let data_sets = make_datasets(|i| make_aos5_blob(i, n));
        let mut result = vec![Cluster::splat(0.0); n];
        bench(b, &data_sets, &mut result, |data_set, result| {
            black_box(compute_aos_blob5(data_set, result));
        });
    }

    #[bench]
    fn bench_5small_blob_aos(b: &mut Bencher) {
        bench_aos_blob5_impl(b, N_SMALL);
    }

    #[bench]
    fn bench_5big_blob_aos(b: &mut Bencher) {
        bench_aos_blob5_impl(b, N_BIG);
    }

    #[test]
    fn test_5_benchmarks() {
        let aos5 = make_aos5(0, 10);
        let soa5 = make_soa5(0, 10);
        let aosblob5 = make_aos5_blob(0, 10);
        let mut result = vec![Cluster::splat(0.0); 10];
        assert_eq!(
            compute_aos5(&aos5, &mut result),
            compute_soa5(&soa5, &mut result)
        );
        assert_eq!(
            compute_aos5(&aos5, &mut result),
            compute_aos_blob5(&aosblob5, &mut result)
        );
    }

    // ------------------------------------------------------------------------

    struct S7 {
        a: Cluster,
        b: Cluster,
        c: Cluster,
        d: Cluster,
        e: Cluster,
        f: Cluster,
        g: Cluster,
    }

    fn make_aos7(seed: usize, n: usize) -> Vec<S7> {
        let mut v = Vec::with_capacity(n);
        for i in seed..seed + n {
            v.push(S7 {
                a: Cluster::splat(i as f32),
                b: Cluster::splat((i / 3) as f32),
                c: Cluster::splat((i * 2) as f32),
                d: Cluster::splat((i * 3) as f32),
                e: Cluster::splat((i * 4) as f32),
                f: Cluster::splat((i + 5) as f32),
                g: Cluster::splat((i * 6) as f32),
            });
        }
        v
    }

    fn compute_aos7(data_set: &Vec<S7>, result: &mut [Cluster]) -> Cluster {
        let mut sum = Cluster::splat(0.0);
        for (s7, r) in data_set.iter().zip(result.iter_mut()) {
            let tmp =
                s7.a.mul_add(s7.b, s7.c)
                    .mul_add(s7.d, s7.e)
                    .mul_add(s7.f, s7.g);
            *r = tmp;
            sum += tmp;
        }
        sum
    }

    fn bench_aos7_impl(b: &mut Bencher, n: usize) {
        let data_sets = make_datasets(|i| make_aos7(i, n));
        let mut result = vec![Cluster::splat(0.0); n];
        bench(b, &data_sets, &mut result, |data_set, result| {
            black_box(compute_aos7(data_set, result));
        });
    }

    #[bench]
    fn bench_7small_aos(b: &mut Bencher) {
        bench_aos7_impl(b, N_SMALL);
    }

    #[bench]
    fn bench_7big_aos(b: &mut Bencher) {
        bench_aos7_impl(b, N_BIG);
    }

    // ------------------------------------------------------------------------
    struct SoA7 {
        a: Vec<Cluster>,
        b: Vec<Cluster>,
        c: Vec<Cluster>,
        d: Vec<Cluster>,
        e: Vec<Cluster>,
        f: Vec<Cluster>,
        g: Vec<Cluster>,
    }

    fn make_soa7(seed: usize, n: usize) -> SoA7 {
        let mut c = Vec::with_capacity(n);
        let mut b = Vec::with_capacity(n);
        let mut e = Vec::with_capacity(n);
        let mut d = Vec::with_capacity(n);
        let mut a = Vec::with_capacity(n);
        let mut f = Vec::with_capacity(n);
        let mut g = Vec::with_capacity(n);
        for i in seed..seed + n {
            a.push(Cluster::splat(i as f32));
            b.push(Cluster::splat((i / 3) as f32));
            c.push(Cluster::splat((i * 2) as f32));
            d.push(Cluster::splat((i * 3) as f32));
            e.push(Cluster::splat((i * 4) as f32));
            f.push(Cluster::splat((i + 5) as f32));
            g.push(Cluster::splat((i * 6) as f32));
        }
        SoA7 {
            a,
            b,
            c,
            d,
            e,
            f,
            g,
        }
    }

    fn bench_soa7_impl(b: &mut Bencher, n: usize) {
        let data_sets = make_datasets(|i| make_soa7(i, n));
        let mut result = vec![Cluster::splat(0.0); n];
        bench(b, &data_sets, &mut result, |data_set, result| {
            black_box(compute_soa7(data_set, result));
        });
    }

    fn compute_soa7(data_set: &SoA7, result: &mut [Cluster]) -> Cluster {
        let mut sum = Cluster::splat(0.0);
        let n = data_set.a.len();
        for i in 0..n {
            let tmp = data_set.a[i]
                .mul_add(data_set.b[i], data_set.c[i])
                .mul_add(data_set.d[i], data_set.e[i])
                .mul_add(data_set.f[i], data_set.g[i]);
            result[i] = tmp;
            sum += tmp;
        }
        sum
    }

    #[bench]
    fn bench_7small_soa(b: &mut Bencher) {
        bench_soa7_impl(b, N_SMALL);
    }

    #[bench]
    fn bench_7big_soa(b: &mut Bencher) {
        bench_soa7_impl(b, N_BIG);
    }

    #[test]
    fn test_7_benchmarks() {
        let aos7 = make_aos7(0, 10);
        let soa7 = make_soa7(0, 10);
        let mut result = vec![Cluster::splat(0.0); 10];
        assert_eq!(
            compute_aos7(&aos7, &mut result),
            compute_soa7(&soa7, &mut result)
        );
    }

    // ------------------------------------------------------------------------

    fn compute_aos7_sparse(data_set: &Vec<S7>, result: &mut [Cluster]) -> Cluster {
        let mut sum = Cluster::splat(0.0);
        for (s7, r) in data_set.iter().zip(result.iter_mut()) {
            let tmp = s7.a.mul_add(s7.c, s7.d).mul(s7.g);
            *r = tmp;
            sum += tmp;
        }
        sum
    }

    fn bench_aos7_sparse_impl(b: &mut Bencher, n: usize) {
        let data_sets = make_datasets(|i| make_aos7(i, n));
        let mut result = vec![Cluster::splat(0.0); n];
        bench(b, &data_sets, &mut result, |data_set, result| {
            black_box(compute_aos7_sparse(data_set, result));
        });
    }

    #[bench]
    fn bench_sparse_7small_aos(b: &mut Bencher) {
        bench_aos7_sparse_impl(b, N_SMALL);
    }

    #[bench]
    fn bench_sparse_7big_aos(b: &mut Bencher) {
        bench_aos7_sparse_impl(b, N_BIG);
    }

    // ------------------------------------------------------------------------

    fn bench_soa7_sparse_impl(b: &mut Bencher, n: usize) {
        let data_sets = make_datasets(|i| make_soa7(i, n));
        let mut result = vec![Cluster::splat(0.0); n];
        bench(b, &data_sets, &mut result, |data_set, result| {
            black_box(compute_soa7_sparse(data_set, result));
        });
    }

    fn compute_soa7_sparse(data_set: &SoA7, result: &mut [Cluster]) -> Cluster {
        let mut sum = Cluster::splat(0.0);
        let n = data_set.a.len();
        for i in 0..n {
            let tmp = data_set.a[i]
                .mul_add(data_set.c[i], data_set.d[i])
                .mul(data_set.g[i]);
            result[i] = tmp;
            sum += tmp;
        }
        sum
    }

    #[bench]
    fn bench_sparse_7small_soa(b: &mut Bencher) {
        bench_soa7_sparse_impl(b, N_SMALL);
    }

    #[bench]
    fn bench_sparse_7big_soa(b: &mut Bencher) {
        bench_soa7_sparse_impl(b, N_BIG);
    }

    #[test]
    fn test_7_sparse_benchmarks() {
        let aos7 = make_aos7(0, 10);
        let soa7 = make_soa7(0, 10);
        let mut result = vec![Cluster::splat(0.0); 10];
        assert_eq!(
            compute_aos7_sparse(&aos7, &mut result),
            compute_soa7_sparse(&soa7, &mut result)
        );
    }

    // ------------------------------------------------------------------------
    struct S8 {
        a: Cluster,
        b: Cluster,
        c: Cluster,
        d: Cluster,
        e: Cluster,
        f: Cluster,
        g: Cluster,
        h: Cluster,
    }

    fn make_aos8(seed: usize, n: usize) -> Vec<S8> {
        let mut v = Vec::with_capacity(n);
        for i in seed..seed + n {
            v.push(S8 {
                a: Cluster::splat(i as f32),
                b: Cluster::splat((i / 3) as f32),
                c: Cluster::splat((i * 2) as f32),
                d: Cluster::splat((i * 3) as f32),
                e: Cluster::splat((i * 4) as f32),
                f: Cluster::splat((i + 5) as f32),
                g: Cluster::splat((i * 6) as f32),
                h: Cluster::splat((i * 7) as f32),
            });
        }
        v
    }

    fn compute_aos8(data_set: &Vec<S8>, result: &mut [Cluster]) -> Cluster {
        let mut sum = Cluster::splat(0.0);
        for (s8, r) in data_set.iter().zip(result.iter_mut()) {
            let tmp =
                s8.a.mul_add(s8.b, s8.c)
                    .mul_add(s8.d, s8.e)
                    .mul_add(s8.f, s8.g)
                    .mul(s8.h);
            *r = tmp;
            sum += tmp;
        }
        sum
    }

    fn bench_aos8_impl(b: &mut Bencher, n: usize) {
        let data_sets = make_datasets(|i| make_aos8(i, n));
        let mut result = vec![Cluster::splat(0.0); n];
        bench(b, &data_sets, &mut result, |data_set, result| {
            black_box(compute_aos8(data_set, result));
        });
    }

    #[bench]
    fn bench_8small_aos(b: &mut Bencher) {
        bench_aos8_impl(b, N_SMALL);
    }

    #[bench]
    fn bench_8big_aos(b: &mut Bencher) {
        bench_aos8_impl(b, N_BIG);
    }

    // ------------------------------------------------------------------------

    struct SoA8 {
        a: Vec<Cluster>,
        b: Vec<Cluster>,
        c: Vec<Cluster>,
        d: Vec<Cluster>,
        e: Vec<Cluster>,
        f: Vec<Cluster>,
        g: Vec<Cluster>,
        h: Vec<Cluster>,
    }

    fn make_soa8(seed: usize, n: usize) -> SoA8 {
        let mut c = Vec::with_capacity(n);
        let mut b = Vec::with_capacity(n);
        let mut e = Vec::with_capacity(n);
        let mut d = Vec::with_capacity(n);
        let mut a = Vec::with_capacity(n);
        let mut f = Vec::with_capacity(n);
        let mut h = Vec::with_capacity(n);
        let mut g = Vec::with_capacity(n);
        for i in seed..seed + n {
            a.push(Cluster::splat(i as f32));
            b.push(Cluster::splat((i / 3) as f32));
            c.push(Cluster::splat((i * 2) as f32));
            d.push(Cluster::splat((i * 3) as f32));
            e.push(Cluster::splat((i * 4) as f32));
            f.push(Cluster::splat((i + 5) as f32));
            g.push(Cluster::splat((i * 6) as f32));
            h.push(Cluster::splat((i * 7) as f32));
        }
        SoA8 {
            a,
            b,
            c,
            d,
            e,
            f,
            g,
            h,
        }
    }

    fn bench_soa8_impl(b: &mut Bencher, n: usize) {
        let data_sets = make_datasets(|i| make_soa8(i, n));
        let mut result = vec![Cluster::splat(0.0); n];
        bench(b, &data_sets, &mut result, |data_set, result| {
            black_box(compute_soa8(data_set, result));
        });
    }

    fn compute_soa8(data_set: &SoA8, result: &mut [Cluster]) -> Cluster {
        let mut sum = Cluster::splat(0.0);
        let n = data_set.a.len();
        for i in 0..n {
            let tmp = data_set.a[i]
                .mul_add(data_set.b[i], data_set.c[i])
                .mul_add(data_set.d[i], data_set.e[i])
                .mul_add(data_set.f[i], data_set.g[i])
                .mul(data_set.h[i]);
            result[i] = tmp;
            sum += tmp;
        }
        sum
    }

    #[bench]

    fn bench_8small_soa(b: &mut Bencher) {
        bench_soa8_impl(b, N_SMALL);
    }

    #[bench]
    fn bench_8big_soa(b: &mut Bencher) {
        bench_soa8_impl(b, N_BIG);
    }

    // ------------------------------------------------------------------------
    struct AoSblob8 {
        data: Vec<Cluster>,
    }

    fn make_aos8_blob(seed: usize, n: usize) -> AoSblob8 {
        let mut data = Vec::with_capacity(n * 8);
        for i in seed..seed + n {
            data.push(Cluster::splat(i as f32));
            data.push(Cluster::splat((i / 3) as f32));
            data.push(Cluster::splat((i * 2) as f32));
            data.push(Cluster::splat((i * 3) as f32));
            data.push(Cluster::splat((i * 4) as f32));
            data.push(Cluster::splat((i + 5) as f32));
            data.push(Cluster::splat((i * 6) as f32));
            data.push(Cluster::splat((i * 7) as f32));
        }
        AoSblob8 { data }
    }

    fn compute_aos_blob8(data_set: &AoSblob8, result: &mut [Cluster]) -> Cluster {
        let mut sum = Cluster::splat(0.0);
        let mut i = 0;
        for r in result.iter_mut() {
            let tmp = data_set.data[i]
                .mul_add(data_set.data[i + 1], data_set.data[i + 2])
                .mul_add(data_set.data[i + 3], data_set.data[i + 4])
                .mul_add(data_set.data[i + 5], data_set.data[i + 6])
                .mul(data_set.data[i + 7]);
            *r = tmp;
            sum += tmp;
            i += 8;
        }
        sum
    }

    fn bench_aos_blob8_impl(b: &mut Bencher, n: usize) {
        let data_sets = make_datasets(|i| make_aos8_blob(i, n));
        let mut result = vec![Cluster::splat(0.0); n];
        bench(b, &data_sets, &mut result, |data_set, result| {
            black_box(compute_aos_blob8(data_set, result));
        });
    }

    #[bench]
    fn bench_8small_blob_aos(b: &mut Bencher) {
        bench_aos_blob8_impl(b, N_SMALL);
    }

    #[bench]
    fn bench_8big_blob_aos(b: &mut Bencher) {
        bench_aos_blob8_impl(b, N_BIG);
    }

    #[test]
    fn test_8_benchmarks() {
        let aos8 = make_aos8(0, 10);
        let soa8 = make_soa8(0, 10);
        let aosblob8 = make_aos8_blob(0, 10);
        let mut result = vec![Cluster::splat(0.0); 10];
        assert_eq!(
            compute_aos8(&aos8, &mut result),
            compute_soa8(&soa8, &mut result)
        );
        assert_eq!(
            compute_aos8(&aos8, &mut result),
            compute_aos_blob8(&aosblob8, &mut result)
        );
    }

    // ------------------------------------------------------------------------
    struct S9 {
        a: Cluster,
        b: Cluster,
        c: Cluster,
        d: Cluster,
        e: Cluster,
        f: Cluster,
        g: Cluster,
        h: Cluster,
        i: Cluster,
    }

    fn make_aos9(seed: usize, n: usize) -> Vec<S9> {
        let mut v = Vec::with_capacity(n);
        for i in seed..seed + n {
            v.push(S9 {
                a: Cluster::splat(i as f32),
                b: Cluster::splat((i / 3) as f32),
                c: Cluster::splat((i * 2) as f32),
                d: Cluster::splat((i * 3) as f32),
                e: Cluster::splat((i * 4) as f32),
                f: Cluster::splat((i + 5) as f32),
                g: Cluster::splat((i * 6) as f32),
                h: Cluster::splat((i * 7) as f32),
                i: Cluster::splat((i * 8) as f32),
            });
        }
        v
    }

    fn compute_aos9(data_set: &Vec<S9>, result: &mut [Cluster]) -> Cluster {
        let mut sum = Cluster::splat(0.0);
        for (s9, r) in data_set.iter().zip(result.iter_mut()) {
            let tmp =
                s9.a.mul_add(s9.b, s9.c)
                    .mul_add(s9.d, s9.e)
                    .mul_add(s9.f, s9.g)
                    .mul_add(s9.h, s9.i);
            *r = tmp;
            sum += tmp;
        }
        sum
    }

    fn bench_aos9_impl(b: &mut Bencher, n: usize) {
        let data_sets = make_datasets(|i| make_aos9(i, n));
        let mut result = vec![Cluster::splat(0.0); n];
        bench(b, &data_sets, &mut result, |data_set, result| {
            black_box(compute_aos9(data_set, result));
        });
    }

    #[bench]

    fn bench_9small_aos(b: &mut Bencher) {
        bench_aos9_impl(b, N_SMALL);
    }

    #[bench]
    fn bench_9big_aos(b: &mut Bencher) {
        bench_aos9_impl(b, N_BIG);
    }

    // ------------------------------------------------------------------------
    struct SoA9 {
        a: Vec<Cluster>,
        b: Vec<Cluster>,
        c: Vec<Cluster>,
        d: Vec<Cluster>,
        e: Vec<Cluster>,
        f: Vec<Cluster>,
        g: Vec<Cluster>,
        h: Vec<Cluster>,
        i: Vec<Cluster>,
    }

    fn make_soa9(seed: usize, n: usize) -> SoA9 {
        let mut c = Vec::with_capacity(n);
        let mut b = Vec::with_capacity(n);
        let mut e = Vec::with_capacity(n);
        let mut d = Vec::with_capacity(n);
        let mut a = Vec::with_capacity(n);
        let mut f = Vec::with_capacity(n);
        let mut i = Vec::with_capacity(n);
        let mut h = Vec::with_capacity(n);
        let mut g = Vec::with_capacity(n);
        for j in seed..seed + n {
            a.push(Cluster::splat(j as f32));
            b.push(Cluster::splat((j / 3) as f32));
            c.push(Cluster::splat((j * 2) as f32));
            d.push(Cluster::splat((j * 3) as f32));
            e.push(Cluster::splat((j * 4) as f32));
            f.push(Cluster::splat((j + 5) as f32));
            g.push(Cluster::splat((j * 6) as f32));
            h.push(Cluster::splat((j * 7) as f32));
            i.push(Cluster::splat((j * 8) as f32));
        }
        SoA9 {
            a,
            b,
            c,
            d,
            e,
            f,
            g,
            h,
            i,
        }
    }

    fn bench_soa9_impl(b: &mut Bencher, n: usize) {
        let data_sets = make_datasets(|i| make_soa9(i, n));
        let mut result = vec![Cluster::splat(0.0); n];
        bench(b, &data_sets, &mut result, |data_set, result| {
            black_box(compute_soa9(data_set, result));
        });
    }

    fn compute_soa9(data_set: &SoA9, result: &mut [Cluster]) -> Cluster {
        let mut sum = Cluster::splat(0.0);
        let n = data_set.a.len();
        for i in 0..n {
            let tmp = data_set.a[i]
                .mul_add(data_set.b[i], data_set.c[i])
                .mul_add(data_set.d[i], data_set.e[i])
                .mul_add(data_set.f[i], data_set.g[i])
                .mul_add(data_set.h[i], data_set.i[i]);
            result[i] = tmp;
            sum += tmp;
        }
        sum
    }

    #[bench]
    fn bench_9small_soa(b: &mut Bencher) {
        bench_soa9_impl(b, N_SMALL);
    }

    #[bench]
    fn bench_9big_soa(b: &mut Bencher) {
        bench_soa9_impl(b, N_BIG);
    }

    // ------------------------------------------------------------------------
    struct AoSblob9 {
        data: Vec<Cluster>,
    }

    fn make_aos9_blob(seed: usize, n: usize) -> AoSblob9 {
        let mut data = Vec::with_capacity(n * 9);
        for i in seed..seed + n {
            data.push(Cluster::splat(i as f32));
            data.push(Cluster::splat((i / 3) as f32));
            data.push(Cluster::splat((i * 2) as f32));
            data.push(Cluster::splat((i * 3) as f32));
            data.push(Cluster::splat((i * 4) as f32));
            data.push(Cluster::splat((i + 5) as f32));
            data.push(Cluster::splat((i * 6) as f32));
            data.push(Cluster::splat((i * 7) as f32));
            data.push(Cluster::splat((i * 8) as f32));
        }
        AoSblob9 { data }
    }

    fn compute_aos_blob9(data_set: &AoSblob9, result: &mut [Cluster]) -> Cluster {
        let mut sum = Cluster::splat(0.0);
        let mut i = 0;
        for r in result.iter_mut() {
            let tmp = data_set.data[i]
                .mul_add(data_set.data[i + 1], data_set.data[i + 2])
                .mul_add(data_set.data[i + 3], data_set.data[i + 4])
                .mul_add(data_set.data[i + 5], data_set.data[i + 6])
                .mul_add(data_set.data[i + 7], data_set.data[i + 8]);
            *r = tmp;
            sum += tmp;
            i += 9;
        }
        sum
    }

    fn bench_aos_blob9_impl(b: &mut Bencher, n: usize) {
        let data_sets = make_datasets(|i| make_aos9_blob(i, n));
        let mut result = vec![Cluster::splat(0.0); n];
        bench(b, &data_sets, &mut result, |data_set, result| {
            black_box(compute_aos_blob9(data_set, result));
        });
    }

    #[bench]
    fn bench_9small_blob_aos(b: &mut Bencher) {
        bench_aos_blob9_impl(b, N_SMALL);
    }

    #[bench]
    fn bench_9big_blob_aos(b: &mut Bencher) {
        bench_aos_blob9_impl(b, N_BIG);
    }

    #[test]
    fn test_9_benchmarks() {
        let aos9 = make_aos9(0, 10);
        let soa9 = make_soa9(0, 10);
        let aosblob9 = make_aos9_blob(0, 10);
        let mut result = vec![Cluster::splat(0.0); 10];
        let tmp = compute_aos9(&aos9, &mut result);
        assert_eq!(tmp, compute_soa9(&soa9, &mut result));
        assert_eq!(tmp, compute_aos_blob9(&aosblob9, &mut result));
    }
    // ------------------------------------------------------------------------

    fn compute_sparse_aos_blob9(data_set: &AoSblob9, result: &mut [Cluster]) -> Cluster {
        let mut sum = Cluster::splat(0.0);
        let mut i = 0;
        for r in result.iter_mut() {
            let tmp = data_set.data[i]
                .mul(data_set.data[i + 2])
                .mul_add(data_set.data[i + 5], data_set.data[i + 6])
                .mul_add(data_set.data[i + 7], data_set.data[i + 8]);
            *r = tmp;
            sum += tmp;
            i += 9;
        }
        sum
    }

    fn bench_sparse_aos_blob9_impl(b: &mut Bencher, n: usize) {
        let data_sets = make_datasets(|i| make_aos9_blob(i, n));
        let mut result = vec![Cluster::splat(0.0); n];
        bench(b, &data_sets, &mut result, |data_set, result| {
            black_box(compute_sparse_aos_blob9(data_set, result));
        });
    }

    #[bench]
    fn bench_sparse_9small_blob_aos(b: &mut Bencher) {
        bench_sparse_aos_blob9_impl(b, N_SMALL);
    }

    #[bench]
    fn bench_sparse_9big_blob_aos(b: &mut Bencher) {
        bench_sparse_aos_blob9_impl(b, N_BIG);
    }

    // ------------------------------------------------------------------------

    fn compute_sparse_soa9(data_set: &SoA9, result: &mut [Cluster]) -> Cluster {
        let mut sum = Cluster::splat(0.0);
        let n = data_set.a.len();
        for i in 0..n {
            let tmp = data_set.a[i]
                .mul(data_set.c[i])
                .mul_add(data_set.f[i], data_set.g[i])
                .mul_add(data_set.h[i], data_set.i[i]);
            result[i] = tmp;
            sum += tmp;
        }
        sum
    }

    fn bench_sparse_soa9_impl(b: &mut Bencher, n: usize) {
        let data_sets = make_datasets(|i| make_soa9(i, n));
        let mut result = vec![Cluster::splat(0.0); n];
        bench(b, &data_sets, &mut result, |data_set, result| {
            black_box(compute_sparse_soa9(data_set, result));
        });
    }

    #[bench]
    fn bench_sparse_9small_soa(b: &mut Bencher) {
        bench_sparse_soa9_impl(b, N_SMALL);
    }

    #[bench]
    fn bench_sparse_9big_soa(b: &mut Bencher) {
        bench_sparse_soa9_impl(b, N_BIG);
    }

    #[test]
    fn test_sparse_9_benchmarks() {
        let aos9 = make_aos9_blob(0, 10);
        let soa9 = make_soa9(0, 10);
        let mut result = vec![Cluster::splat(0.0); 10];
        assert_eq!(
            compute_sparse_aos_blob9(&aos9, &mut result),
            compute_sparse_soa9(&soa9, &mut result)
        );
    }
}
