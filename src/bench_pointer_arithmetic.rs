#[cfg(test)]
mod tests {
    use crate::utils::{bench, Data};
    use std::simd::{num::SimdFloat, StdFloat};
    use test::black_box;

    use crate::utils::{make_data, make_datasets, ArgsIn, ArgsOut, Cluster};

    fn f_slices(args_in: &ArgsIn, args_out: &mut ArgsOut) {
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

    fn runtime_slices(data_in: &Data, data_out: &mut Data, f: fn(&ArgsIn, &mut ArgsOut)) -> f32 {
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
    fn bench_index_slices(b: &mut test::Bencher) {
        let data_in = make_datasets(|_| (make_data()));
        let mut data_out = make_datasets(|_| (make_data()));
        bench(b, &data_in, &mut data_out, |d_in, d_out| {
            black_box(runtime_slices(d_in, d_out, f_slices));
        });
    }

    // ------------------------------------------------------------------------

    fn f_pointer_arithmetics(
        mut a_ptr: *const Cluster,
        mut c_ptr: *const Cluster,
        mut d_ptr: *const Cluster,
        mut e_ptr: *const Cluster,
        mut b_ptr: *mut Cluster,
        a_step: usize,
        c_step: usize,
        d_step: usize,
        e_step: usize,
        b_step: usize,
        len: usize,
    ) {
        let mut sum = Cluster::splat(0.0);
        for _ in 0..len {
            {
                let a = unsafe { *a_ptr };
                let c = unsafe { *c_ptr };
                let d = unsafe { *d_ptr };
                let e = unsafe { *e_ptr };
                let b: &mut Cluster = unsafe { &mut *b_ptr };

                let tmp = a.mul_add(c.sqrt(), d / e);
                sum += tmp;
                *b = sum;
            }

            a_ptr = unsafe { a_ptr.byte_offset(a_step as isize) };
            c_ptr = unsafe { c_ptr.byte_offset(c_step as isize) };
            d_ptr = unsafe { d_ptr.byte_offset(d_step as isize) };
            e_ptr = unsafe { e_ptr.byte_offset(e_step as isize) };
            b_ptr = unsafe { b_ptr.byte_offset(b_step as isize) };
        }
    }

    fn runtime_ptr_arithmetics(
        data_in: &Data,
        data_out: &mut Data,
        f: fn(
            a: *const Cluster,
            c: *const Cluster,
            d: *const Cluster,
            e: *const Cluster,
            b: *mut Cluster,
            a_step: usize,
            c_step: usize,
            d_step: usize,
            e_step: usize,
            b_step: usize,
            len: usize,
        ),
    ) -> f32 {
        let a = data_in.a.as_ptr();
        let c = data_in.c.as_ptr();
        let d = data_in.d.as_ptr();
        let e = data_in.e.as_ptr();
        let b = data_out.b.as_mut_ptr();
        let len = data_in.a.len();

        f(
            a,
            c,
            d,
            e,
            b,
            size_of::<Cluster>(),
            size_of::<Cluster>(),
            size_of::<Cluster>(),
            size_of::<Cluster>(),
            size_of::<Cluster>(),
            len,
        );

        unsafe { b.add(len - 1).read() }.reduce_sum()
    }

    #[bench]
    fn bench_index_ptr_arithmetics(b: &mut test::Bencher) {
        let data_in = make_datasets(|_| (make_data()));
        let mut data_out = make_datasets(|_| (make_data()));
        bench(b, &data_in, &mut data_out, |d_in, d_out| {
            black_box(runtime_ptr_arithmetics(d_in, d_out, f_pointer_arithmetics));
        });
    }

    // ------------------------------------------------------------------------
    #[test]
    fn test_index() {
        let data_in = make_data();
        let mut data_out1 = make_data();
        let mut data_out2 = make_data();
        let sum1 = runtime_slices(&data_in, &mut data_out1, f_slices);
        let sum2 = runtime_ptr_arithmetics(&data_in, &mut data_out2, f_pointer_arithmetics);
        assert_eq!(sum1, sum2);
    }
}
