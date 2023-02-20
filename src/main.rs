use ndarray::linalg::Dot;
use ndarray::prelude::*;
use num::{Complex, Zero};
//use std::iter::Sum;
use std::ops::Mul;

pub trait Phasor {
    fn phasor(arg: i32, m: usize) -> Self;
}

impl Phasor for Complex<f32> {
    fn phasor(arg: i32, n: usize) -> Self {
        use std::f32::consts::PI;
        Complex::new(0.0, -2.0 * PI * arg as f32 / (n as f32)).exp()
    }
}

pub fn dft(y: &Array1<Complex<f32>>) -> Array1<Complex<f32>> {
    let mut x = y.clone();
    let n = x.len();
    for m in 0..n {
        let w = Array::from_iter((0..n).map(|k| {
            let a: Complex<f32> = Phasor::phasor((k * m) as i32, n);
            a
        }));
        let v = w.dot(y);
        x[m] = v;
    }
    x
}

pub fn xdft<T>(y: &Array1<T>) -> Array1<T>
where
    T: Phasor + Clone + Copy + Mul + Mul<Output = T>,
    <T as Mul>::Output: Clone + Copy + Zero,
{
    let mut x = y.clone();
    let n = x.len();
    for m in 0..n {
        let w = Array::from_iter((0..n).map(|k| {
            let a: T = Phasor::phasor((k * m) as i32, n);
            a * y[m]
        }))
        .sum();

        x[m] = w;
    }
    x
}

pub fn ydft<T>(y: &Array1<T>) -> Array1<T>
where
    T: Phasor + Clone + Copy + Dot<Array1<T>, Output = T>,
    dyn Dot<Array1<T>, Output = T>: Copy + Clone,
    <T as Dot<Array1<T>>>::Output: Copy + Clone,
    Array1<T>: Dot<Array1<T>>,
{
    let mut x = y.clone();
    let n = x.len();
    for m in 0..n {
        let w = Array::from_iter((0..n).map(|k| {
            let a: T = Phasor::phasor((k * m) as i32, n);
            a
        }));

        x[m] = y.dot(&w);
    }
    x
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn first() {
        let a = arr1(&[1.0, 1.0, 1.0]).mapv(|x| Complex::new(x, 0.0));
        let v = &dft(&a);
        dbg!(v);
        let z = &arr1(&[3.0, 0.0, 0.0]).mapv(|x| Complex::<f32>::new(x, 0.0));
        for i in 0..3 {
            let a0 = v[i] * v[i];
            let a1 = z[i] * z[i];
            assert!(a0.re - a1.re < 1e-4);
            assert!(a0.im - a1.im < 1e-4);
        }
    }
    #[test]
    fn second() {
        let a = arr1(&[1.0, 1.0, 1.0]).mapv(|x| Complex::new(x, 0.0));
        let v = &xdft(&a);
        dbg!(v);
        let z = &arr1(&[3.0, 0.0, 0.0]).mapv(|x| Complex::<f32>::new(x, 0.0));
        for i in 0..3 {
            let a0 = v[i] * v[i];
            let a1 = z[i] * z[i];
            assert!(a0.re - a1.re < 1e-4);
            assert!(a0.im - a1.im < 1e-4);
        }
    }
    #[test]
    fn third() {
        let a = arr1(&[1.0, 1.0, 1.0]).mapv(|x| Complex::new(x, 0.0));
        let v = &ydft(&a);
        dbg!(v);
        let z = &arr1(&[3.0, 0.0, 0.0]).mapv(|x| Complex::<f32>::new(x, 0.0));
        for i in 0..3 {
            let a0 = v[i] * v[i];
            let a1 = z[i] * z[i];
            assert!(a0.re - a1.re < 1e-4);
            assert!(a0.im - a1.im < 1e-4);
        }
    }
}

fn main() {
    println!("hello!");
}
