# Cyclotomic Integer Approximation

This repository contains testing code used for the work

>
> William H. Pan, Dylan Roscow, and Netanel Raviv. Complex sensing: Reconstructing dense rational signals from few complex measurements. IEEE Signal Processing Letters (in-press).

In the project we encountered the following subproblem.
Take $n^\text{th}$ roots of unity $\zeta_n^k = (e^{(2\pi / n) i})^k$. 
Given a "cyclotomic integer" $z$, that is, a sum of $n^\text{th}$ roots of unity

$z = c_0\zeta_n^0 + c_1\zeta_n^1 + \cdots + c_{n - 1}\zeta_n^{n - 1}$

where $c_0, c_1, \ldots, c_{n - 1}$ are unknown integers, can we find or at least approximate the $c_0, c_1, \ldots, c_{n - 1}$? 
To our knowledge there exists no analytical method.
In this code we implement the LLL algorithm \[Lenstra, Lenstra, Lovász (1982)\] to recover $c_0, c_1, \ldots, c_{n - 1}$ and test its success rate against various random distirbutions over the cyclotomic integers.
We also include an implementation and attempt to generalize an approximation algorithm for complex numbers by cyclotomic integers \[Shokrollahi & Stemann (1996)\] arising from $n$ that are powers of 2.

For questions reach out to whpan \[at\] utexas \[dot\] edu or droscow \[at\] purdue \[dot\] edu.

## Instructions

The code uses NumPy, [gmpy2](https://gmpy2.readthedocs.io/en/latest/), [fpylll](https://github.com/fplll/fpylll), and [SageMath](https://doc.sagemath.org/html/en/installation/).

`lll.py` contains all helper functions used in testing. 
We plot testing data in `examples.ipynb`, which any user should feel free to experiment with.
The main function to be used is `run_trials` which takes as parameters
- `orders`: the list of `n`'s to test against.
- `gen` and `gen_param`: selection of the random generator and its parameter.
- `Ascale` and `Bscale`: hyperparameters for our LLL matrix. In the noiseless case, highest success rates are achieved when `Ascale` is to the highest allowed by `prec` and `Bscale` is set around 3. For the noisy case set `Bscale` to 0 to use our formula to approximate the optimal `Bscale` value.
- `last_root`: boolean setting whether to include $\zeta_n^{n - 1}$.
- `num_trials`: number of trials.
- `noise`: variance of random Gaussian noise to be added to the generated random cyclotomic integer before processing.
- `seed`: set random generator seed.
- `prec`: number of bits to use to define numbers. Implemented with `gmpy2.mpfr`.

For more guidance on best values to use see the testing data in `testing.xlsx`. 

`approximation.ipynb` includes functions to implement the aforementioned direct approximation algorithm and example usage.
