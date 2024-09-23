use std::simd::Simd;

use test::Bencher;

pub type Cluster = Simd<f32, 8>;

/// number of datasets
pub const M: usize = 300;

pub type Step = usize;

pub const N_ARGS_IN: usize = 8;
pub const N_ARGS_OUT: usize = 8;
pub const N: usize = 256;

pub type ArgsIn<'a> = [(&'a [Cluster], Step); N_ARGS_IN];
pub type ArgsOut<'a> = [(&'a mut [Cluster], Step); N_ARGS_OUT];

pub struct Data {
    pub a: Vec<Cluster>,
    pub b: Vec<Cluster>,
    pub c: Vec<Cluster>,
    pub d: Vec<Cluster>,
    pub e: Vec<Cluster>,
}

pub fn make_data() -> Data {
    Data {
        b: vec![Cluster::splat(1.0); N],
        d: vec![Cluster::splat(2.0); N],
        c: vec![Cluster::splat(3.0); N],
        a: vec![Cluster::splat(4.0); N],
        e: vec![Cluster::splat(5.0); N],
    }
}

pub fn make_datasets<T>(f: impl Fn(usize) -> T) -> Vec<T> {
    let mut v = Vec::with_capacity(M);
    for seed in 0..M {
        v.push(f(seed));
    }
    v
}

#[inline(always)]
pub fn bench(
    b: &mut Bencher,
    data_sets_in: &[Data],
    data_sets_out: &mut [Data],
    mut f: impl FnMut(&Data, &mut Data),
) {
    let mut i = 0;
    b.iter(|| {
        let data_set_in = &data_sets_in[i];
        let data_set_out = &mut data_sets_out[i];
        f(data_set_in, data_set_out);
        i += 7;
        if i >= N {
            i -= N;
        }
    });
}
