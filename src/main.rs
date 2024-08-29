#![feature(portable_simd)]
#![feature(test)]
#[deny(soft_unstable)]
extern crate test;

fn main() {
    println!("Hello, world!");
}

#[cfg(test)]
mod tests {
    use std::{
        hint::black_box,
        simd::{Simd, StdFloat},
    };
    use test::Bencher;

    pub type Cluster = Simd<f32, 8>;

    /// data size
    const N_SMALL: usize = 10;

    /// data size
    const N_BIG: usize = 256;

    /// number of datasets
    const M: usize = 300;

    fn make_datasets<T>(f: impl Fn(usize) -> T) -> Vec<T> {
        let mut v = Vec::with_capacity(M);
        for seed in 0..M {
            v.push(f(seed));
        }
        v
    }

    #[inline(always)]
    fn bench<T>(b: &mut Bencher, data_sets: &[T], f: impl Fn(&T)) {
        let mut i = 0;
        b.iter(|| {
            let data_set = &data_sets[i];
            black_box(f(data_set));
            i += 7;
            if i >= M {
                i = 0;
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

    fn compute_aos3(data_set: &Vec<S3>) -> Cluster {
        let mut sum = Cluster::splat(0.0);
        for s3 in data_set {
            sum += s3.a.mul_add(s3.b, s3.c);
        }
        sum
    }

    fn bench_aos3_impl(b: &mut Bencher, n: usize) {
        let data_sets = make_datasets(|i| make_aos3(i, n));
        bench(b, &data_sets, |data_set: &Vec<S3>| {
            black_box(compute_aos3(data_set));
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
        bench(b, &data_sets, |data_set: &SoA3| {
            black_box(compute_soa3(data_set));
        });
    }

    fn compute_soa3(data_set: &SoA3) -> Cluster {
        let mut sum = Cluster::splat(0.0);
        let n = data_set.a.len();
        for i in 0..n {
            sum += data_set.a[i].mul_add(data_set.b[i], data_set.c[i]);
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
        assert_eq!(compute_aos3(&aos3), compute_soa3(&soa3));
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

    fn compute_aos5(data_set: &Vec<S5>) -> Cluster {
        let mut sum = Cluster::splat(0.0);
        for s5 in data_set {
            sum += s5.a.mul_add(s5.b, s5.c).mul_add(s5.d, s5.e);
        }
        sum
    }

    fn bench_aos5_impl(b: &mut Bencher, n: usize) {
        let data_sets = make_datasets(|i| make_aos5(i, n));
        bench(b, &data_sets, |data_set: &Vec<S5>| {
            black_box(compute_aos5(data_set));
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
        bench(b, &data_sets, |data_set: &SoA5| {
            black_box(compute_soa5(data_set));
        });
    }

    fn compute_soa5(data_set: &SoA5) -> Cluster {
        let mut sum = Cluster::splat(0.0);
        let n = data_set.a.len();
        for i in 0..n {
            sum += data_set.a[i]
                .mul_add(data_set.b[i], data_set.c[i])
                .mul_add(data_set.d[i], data_set.e[i]);
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

    #[test]
    fn test_5_benchmarks() {
        let aos5 = make_aos5(0, 10);
        let soa5 = make_soa5(0, 10);
        assert_eq!(compute_aos5(&aos5), compute_soa5(&soa5));
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

    fn compute_aos7(data_set: &Vec<S7>) -> Cluster {
        let mut sum = Cluster::splat(0.0);
        for s7 in data_set {
            sum +=
                s7.a.mul_add(s7.b, s7.c)
                    .mul_add(s7.d, s7.e)
                    .mul_add(s7.f, s7.g);
        }
        sum
    }

    fn bench_aos7_impl(b: &mut Bencher, n: usize) {
        let data_sets = make_datasets(|i| make_aos7(i, n));
        bench(b, &data_sets, |data_set: &Vec<S7>| {
            black_box(compute_aos7(data_set));
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
        bench(b, &data_sets, |data_set: &SoA7| {
            black_box(compute_soa7(data_set));
        });
    }

    fn compute_soa7(data_set: &SoA7) -> Cluster {
        let mut sum = Cluster::splat(0.0);
        let n = data_set.a.len();
        for i in 0..n {
            sum += data_set.a[i]
                .mul_add(data_set.b[i], data_set.c[i])
                .mul_add(data_set.d[i], data_set.e[i])
                .mul_add(data_set.f[i], data_set.g[i]);
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
        assert_eq!(compute_aos7(&aos7), compute_soa7(&soa7));
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

    fn compute_aos9(data_set: &Vec<S9>) -> Cluster {
        let mut sum = Cluster::splat(0.0);
        for s9 in data_set {
            sum +=
                s9.a.mul_add(s9.b, s9.c)
                    .mul_add(s9.d, s9.e)
                    .mul_add(s9.f, s9.g)
                    .mul_add(s9.h, s9.i);
        }
        sum
    }

    fn bench_aos9_impl(b: &mut Bencher, n: usize) {
        let data_sets = make_datasets(|i| make_aos9(i, n));
        bench(b, &data_sets, |data_set: &Vec<S9>| {
            black_box(compute_aos9(data_set));
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
        bench(b, &data_sets, |data_set: &SoA9| {
            black_box(compute_soa9(data_set));
        });
    }

    fn compute_soa9(data_set: &SoA9) -> Cluster {
        let mut sum = Cluster::splat(0.0);
        let n = data_set.a.len();
        for i in 0..n {
            sum += data_set.a[i]
                .mul_add(data_set.b[i], data_set.c[i])
                .mul_add(data_set.d[i], data_set.e[i])
                .mul_add(data_set.f[i], data_set.g[i])
                .mul_add(data_set.h[i], data_set.i[i]);
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

    #[test]
    fn test_9_benchmarks() {
        let aos9 = make_aos9(0, 10);
        let soa9 = make_soa9(0, 10);
        assert_eq!(compute_aos9(&aos9), compute_soa9(&soa9));
    }
}
