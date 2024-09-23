#[cfg(test)]
pub mod tests {
    use std::{
        env::Args,
        simd::{num::SimdFloat, StdFloat},
        slice,
    };

    use test::{black_box, Bencher};

    use crate::utils::{bench, make_data, make_datasets, ArgsIn, ArgsOut, Cluster, Data, N};

    #[inline]
    fn make_splat_slice(var: &Cluster) -> (&[Cluster], usize) {
        let ptr = var as *const Cluster;
        (unsafe { &slice::from_raw_parts(ptr, 1) }, 0)
    }

    // ------------------------------------------------------------------------

    fn f_singleindexing(args_in: &ArgsIn, args_out: &mut ArgsOut) {
        let a = args_in[0].0;
        let c = args_in[1].0;
        let d = args_in[2].0;
        let e = args_in[3].0;
        let b = args_out[0].0.as_mut();
        let len = a.len();
        let mut sum = Cluster::splat(0.0);
        for i in 0..len {
            let tmp = a[i].mul_add(c[i].sqrt(), d[i] / e[i]);
            sum += tmp;
            b[i] = sum;
        }
    }

    fn runtime(data_in: &Data, data_out: &mut Data, f: fn(&ArgsIn, &mut ArgsOut)) -> f32 {
        let mut args_in = ArgsIn::default();
        args_in[0] = (&data_in.a, 1);
        args_in[1] = (&data_in.c, 1);
        args_in[2] = (&data_in.d, 1);
        args_in[3] = (&data_in.e, 1);

        let mut args_out = ArgsOut::default();
        args_out[0] = (&mut data_out.b, 1);

        f(&args_in, &mut args_out);

        args_out[0].0.last().unwrap().reduce_sum()
    }

    #[bench]
    fn bench_runtime_singleindexing(b: &mut Bencher) {
        let data_in = make_datasets(|_| (make_data()));
        let mut data_out = make_datasets(|_| (make_data()));
        bench(b, &data_in, &mut data_out, |d_in, d_out| {
            black_box(runtime(d_in, d_out, f_singleindexing));
        });
    }
    // ------------------------------------------------------------------------

    fn runtime_with_const_memory(
        data_sets_in: &Vec<Data>,
        data_in: &Data,
        data_out: &mut Data,
        f: fn(&ArgsIn, &mut ArgsOut),
    ) -> f32 {
        let mut args_in = ArgsIn::default();
        args_in[0] = (&*data_sets_in[0].a, 1);
        args_in[1] = (&*data_sets_in[0].c, 1);
        args_in[2] = (&data_in.d, 1);
        args_in[3] = (&data_in.e, 1);

        let mut args_out = ArgsOut::default();
        args_out[0] = (&mut data_out.b, 1);

        f(&args_in, &mut args_out);

        args_out[0].0.last().unwrap().reduce_sum()
    }

    #[bench]
    fn bench_runtime_singleindexing_with_const_memory(b: &mut Bencher) {
        let data_sets_in = make_datasets(|_| (make_data()));
        let mut data_sets_out = make_datasets(|_| (make_data()));

        bench(b, &data_sets_in, &mut data_sets_out, |d_in, d_out| {
            black_box(runtime_with_const_memory(
                &data_sets_in,
                d_in,
                d_out,
                f_singleindexing,
            ));
        });
    }

    // ------------------------------------------------------------------------

    fn f_multiindexing(args_in: &ArgsIn, args_out: &mut ArgsOut) {
        let a = args_in[0].0;
        let c = args_in[1].0;
        let d = args_in[2].0;
        let e = args_in[3].0;
        let b = args_out[0].0.as_mut();

        let len = b.len();
        let mut ia = 0;
        let mut ic = 0;
        let mut id = 0;
        let mut ib = 0;
        let mut ie = 0;

        let mut sum = Cluster::splat(0.0);

        while ib < len {
            let tmp = a[ia].mul_add(c[ic].sqrt(), d[id] / e[ie]);
            sum += tmp;
            b[ib] = sum;
            ia += args_in[0].1;
            ic += args_in[1].1;
            id += args_in[2].1;
            ie += args_in[3].1;
            ib += args_out[0].1;
        }
    }

    fn runtime_with_splats(
        data_in: &Data,
        data_out: &mut Data,
        f: fn(&ArgsIn, &mut ArgsOut),
    ) -> f32 {
        let mut args_in = ArgsIn::default();
        let a = Cluster::splat(4.0);
        let c = Cluster::splat(3.0);
        args_in[0] = make_splat_slice(&a);
        //args_in[0] = (&data_in.a, 1);
        args_in[1] = make_splat_slice(&c);
        args_in[2] = (&data_in.d, 1);
        args_in[3] = (&data_in.e, 1);

        let mut args_out = ArgsOut::default();
        args_out[0] = (&mut data_out.b, 1);

        f(&args_in, &mut args_out);
        args_out[0].0.last().unwrap().reduce_sum()
    }

    #[bench]
    fn bench_runtime_multiindexing_with_splats(b: &mut Bencher) {
        let data_in = make_datasets(|_| (make_data()));
        let mut data_out = make_datasets(|_| (make_data()));
        bench(b, &data_in, &mut data_out, |d_in, d_out| {
            black_box(runtime_with_splats(d_in, d_out, f_multiindexing));
        });
    }

    // ------------------------------------------------------------------------

    #[bench]
    fn bench_runtime_multiindexing(b: &mut Bencher) {
        let data_sets_in = make_datasets(|_| (make_data()));
        let mut data_sets_out = make_datasets(|_| (make_data()));

        bench(b, &data_sets_in, &mut data_sets_out, |d_in, d_out| {
            black_box(runtime(d_in, d_out, f_multiindexing));
        });
    }

    // ------------------------------------------------------------------------
    fn runtime_with_memory_splats(
        data_in: &Data,
        data_out: &mut Data,
        mem1: &mut [Cluster],
        mem2: &mut [Cluster],
        f: fn(&ArgsIn, &mut ArgsOut),
    ) -> f32 {
        let mut args_in = ArgsIn::default();
        let a = Cluster::splat(4.0);
        let c = Cluster::splat(3.0);
        for i in 0..N {
            mem1[i] = a;
            mem2[i] = c;
        }
        args_in[0] = (&mem1, 1);
        args_in[1] = (&mem2, 1);
        args_in[2] = (&data_in.d, 1);
        args_in[3] = (&data_in.e, 1);

        let mut args_out = ArgsOut::default();
        args_out[0] = (&mut data_out.b, 1);

        f(&args_in, &mut args_out);
        args_out[0].0.last().unwrap().reduce_sum()
    }

    #[bench]
    fn bench_runtime_singleindexing_with_memory_splats(b: &mut Bencher) {
        let data_in = make_datasets(|_| (make_data()));
        let mut data_out = make_datasets(|_| (make_data()));
        let mut mem1 = vec![Cluster::splat(0.0); N];
        let mut mem2 = vec![Cluster::splat(0.0); N];
        bench(b, &data_in, &mut data_out, |d_in, d_out| {
            black_box(runtime_with_memory_splats(
                d_in,
                d_out,
                &mut mem1,
                &mut mem2,
                f_singleindexing,
            ));
        });
    }

    #[test]
    fn test_runtime() {
        let data_in = make_datasets(|_| (make_data()));
        let mut data_out = make_data();
        let res1 = runtime(&data_in[0], &mut data_out, f_singleindexing);
        let res2 = runtime(&data_in[0], &mut data_out, f_multiindexing);
        let res3 = runtime_with_splats(&data_in[0], &mut data_out, f_multiindexing);
        let res4 =
            runtime_with_const_memory(&data_in, &data_in[0], &mut data_out, f_singleindexing);
        let res5 = runtime_with_memory_splats(
            &data_in[0],
            &mut data_out,
            &mut vec![Cluster::splat(0.0); N],
            &mut vec![Cluster::splat(0.0); N],
            f_singleindexing,
        );
        assert_eq!(res1, res2);
        assert_eq!(res1, res3);
        assert_eq!(res1, res4);
        assert_eq!(res1, res5);
    }
}
