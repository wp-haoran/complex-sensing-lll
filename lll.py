"""
lll.py
======

LLL-based recovery of cyclotomic integers from few complex measurements.

Given a cyclotomic integer z = c_0 * zeta_p^0 + c_1 * zeta_p^1 + ... + c_{p-1} * zeta_p^{p-1},
where zeta_p is a primitive p-th root of unity and c_0, ..., c_{p-1} are unknown integers,
this module reconstructs the coefficients c_i by reducing the problem to finding a short
vector in an explicitly constructed lattice, solved via the LLL algorithm
(Lenstra, Lenstra, Lovasz 1982) as implemented in `fpylll`.

This code accompanies:
    W. H. Pan, D. Roscow, N. Raviv. "Complex Sensing: Reconstructing Dense Rational
    Signals from Few Complex Measurements." IEEE Signal Processing Letters (in press).

See README.md for usage and `examples.ipynb` for worked examples reproducing the
paper's experimental results.
"""

# Imports for numerical computations
import math
import numpy as np
import gmpy2
from fpylll import IntegerMatrix, LLL

# Utility
import sys
import time
import matplotlib.pyplot as plt
import gc

## Random cyclotomic integer generation functions

def coefficients_gen_norm(
        p: int,
        stdev: float,
        rng: np.random.Generator
    ) -> list[int]:
    """
    Generate random cyclotomic integer summed from `p`th roots of unity
    with each coefficient rounded to an integer from a sample of the Gaussian distribution.

    ### Parameters
    - `p`: order of root of unity.
    - `stdev`: standard deviation of Gaussian distribution.
    - `rng`: NumPy random number generator.

    ### Returns:
    - `coefs`: coefficients of roots of unity in sum, where `coefs[i]` corresponds to the `i`th root.
    """

    return np.rint(rng.normal(loc=0.0, scale=stdev, size=p)).tolist()


def coefficients_gen_l1_pos(
        p: int,
        l1: int,
        rng: np.random.Generator
    ) -> list[int]:
    """
    Generate random cyclotomic integer summed from `p`th roots of unity
    sampled uniformly at random from the positive part of the `l1` sphere.
    
    ### Parameters
    - `p`: order of root of unity.
    - `l1`: sum of coefficients in desired cyclotomic integer.
    - `rng`: NumPy random number generator.
    
    ### Returns:
    - `coefs`: coefficients of roots of unity in sum, where `coefs[i]` corresponds to the `i`th root.
    """

    # Divide coefficients with stars-and-bars approach
    dividers = rng.choice(l1 + 1, size = p - 1, replace=True)
    dividers.sort()

    return np.diff(dividers, prepend=0, append=l1).tolist()


def coefficients_gen_l1(
        p: int, 
        l1: int,
        rng: np.random.Generator
    ) -> list[int]:
    """
    Generate random cyclotomic integer summed from `p`th roots of unity
    sampled uniformly at random from `l1` sphere.
    
    ### Parameters
    - `p`: order of root of unity.
    - `l1`: sum of absolute value of coefficients in desired cyclotomic integer.
    - `rng`: NumPy random number generator.
    
    ### Returns:
    - `coefs`: coefficients of roots of unity in sum, where `coefs[i]` corresponds to the `i`th root.
    """

    # NOTE: coefficients are generated as Python ints (arbitrary precision),
    # so magnitudes beyond 2**64 are supported natively; only downstream use
    # of fixed-width types (e.g. numpy int64 arrays) would impose a ceiling.

    # Uniformly select which coefficients are negative
    negatives = rng.integers(0, 1, size=p, endpoint=True)

    return np.multiply(
        coefficients_gen_l1_pos(p, l1, rng),
        -2 * negatives + 1
    ).tolist()


## Recovery function

def recover_sparse_cyclotomic(
        coefs: list[int], 
        p: int, 
        Ascale: int, 
        Bscale: int, 
        last_root: bool, 
        noise: gmpy2.mpfr, 
        rng: np.random.Generator,
        prec=53
    ) -> IntegerMatrix:
    """
    For a cyclotomic integer `z` summed from `p`th roots of unity,
    generates LLL (short, nearly orthogonal) basis vectors to approximate `z` with integer relations. \n

    Each basis vector `vec` consists of:
    - `vec[0:2]` corresponding to the Re and Im parts of the objective value.
    - `vec[2:p+1+last_root]` corresponding to the coefficients for each root of unity.
    - `vec[-1]` corresponding to the target value.

    ### Parameters:
    - `coefs`: coefficients in cyclotomic integer decomposition of `z`.
    - `p`: order of root of unity.
    - `Ascale`: lattice scaling factor for objective complex values.
    - `Bscale`: lattice scaling factor for number of vectors summed.
    - `last_root`: whether to include `p-1`th root of unity in linear combinations.
    - `noise`: standard deviation of Gaussian noise to add to input cyclotomic integer.
    - `rng`: NumPy random number generator.
    - `prec`: number of precision bits used to define the mantissa. Default '53' employs the IEEE-754 standard for floating numbers.

    ### Returns:
    - `exps`: list of number of roots of unity used in summing z, where `exps[i]` corresponds to the `i`th root.
    - `lll_mat`: list of candidate basis vectors as output of LLL algorithm.
    """

    # Default precision: use numpy.float64 for speed
    if prec == 53:
        # Generate cyclotomic integer
        angles = 2 * np.pi * np.arange(p) / p
        roots_re = np.cos(angles)
        roots_im = np.sin(angles)
        z_re = np.dot(coefs, roots_re)
        z_im = np.dot(coefs, roots_im)

        # Add complex random Gaussian noise with stdev noise
        if noise:
            z_re += rng.normal(scale=noise/math.sqrt(2))
            z_im += rng.normal(scale=noise/math.sqrt(2))
    
    # Arbitrary precision: use gmpy2 wrapper of GMP and MPFR
    else:
        if prec < 1:
            raise ValueError("Precision must be a positive integer.")

        # Set up arbitrary precision environment
        ctx = gmpy2.get_context()
        ctx.precision = prec

        roots_re = [gmpy2.cos((2 * gmpy2.const_pi() * i) / p) for i in range(p)]
        roots_im = [gmpy2.sin((2 * gmpy2.const_pi() * i) / p) for i in range(p)]
        z_re = sum(root_re * coef for root_re, coef in zip(roots_re, coefs))
        z_im = sum(root_im * coef for root_im, coef in zip(roots_im, coefs))

        if noise:
            rs = gmpy2.random_state(int(rng.integers(sys.maxsize)))
            z_re += gmpy2.mpfr_nrandom(rs) * noise/gmpy2.rootn(2, 2)
            z_im += gmpy2.mpfr_nrandom(rs) * noise/gmpy2.rootn(2, 2)

    # Construct the lattice basis matrix
    M = IntegerMatrix(p + last_root, p + 2 + last_root)

    for i in range(p - 1 + last_root):
        M[i, 0] = int(round(roots_re[i] * Ascale)) # roots of unity real parts
        M[i, 1] = int(round(roots_im[i] * Ascale)) # roots of unity imaginary parts
        M[i, 2 + i] = Bscale # identity matrix
    
    # Set target row to real + imaginary parts
    M[-1, 0] = -1 * int(round(z_re * Ascale))
    M[-1, 1] = -1 * int(round(z_im * Ascale))
    M[-1, -1] = 1

    LLL.reduction(M)
    return M


## Testing functions

def run_trials(
        orders: list[int], 
        gen: str,
        gen_param: int, 
        Ascale: int, 
        Bscale: int, 
        last_root: bool, 
        num_trials: int, 
        noise=gmpy2.mpfr(0), 
        seed=None,
        prec=53
    ) -> list[int]:
    """
    Tests success rate of using LLL algorithm to solve
    the cyclotomic integer decomposition integer relation problem.

    ### Parameters:
    - `orders`: list of orders of root of unity to test.
    - `gen`: sets generator for random cyclotomic integers:
    \t - "l1pos" samples cyclotomic integers from positive part of `l1` sphere of radius `gen_param` uniformly at random.
    \t - "l1" samples cyclotomic integers from `l1` sphere of radius `gen_param` uniformly at random.
    \t - "normal" samples coefficients from an unbiased normal distribution with standard deviation `gen_param`.
    - `gen_param`: see above.
    - `Ascale`: lattice scaling factor for objective complex values.
    - `Bscale`: lattice scaling factor for integer mapping. If set to 0:
    \t - in the noiseless case (`noise == 0.0`), `Bscale` defaults to `1`.
    \t - in the noisy case, if `Bscale == 0`, computes optimal `Bscale` for each prime in `primes` for best success rate.
    - `last_root`: whether to include `p-1`th root of unity in linear combinations.
    - `num_trials`: number of trials to run for each `p` in `orders`.
    - `noise`: (optional) standard deviation of Gaussian noise to add to input cyclotomic integer. Default `0.0`.
    - `seed`: (optional) random seed for initializing random number generator.
    - `prec`: (optional) number of precision bits to define float mantissa. Default `53` uses IEEE-754 `numpy.float64`.

    ### Returns:
    - `results`: list of number of successes corresponding to `orders`.
    """

    # Generate placeholder to record correct trials for each prime
    results = [0] * len(orders)

    # Initialize and set random cyclotomic integer generator
    rng = np.random.default_rng(seed)

    if gen == "l1pos":
        generator = coefficients_gen_l1_pos
    elif gen == "l1":
        generator = coefficients_gen_l1
    elif gen == "normal":
        generator = coefficients_gen_norm
    else:
        raise ValueError("Select cyclotomic integer generation mode `l1pos`, `l1`, or `normal`.")
    
    # Run trials for each order in orders
    for i in range(len(orders)):
        p = orders[i]

        # autocomputes Bscale if given noise
        if Bscale == 0 and noise != 0:
            if gen == "l1pos" or gen == "l1":
                Bval = max(int((noise * Ascale * p) // (4.5 * (gen_param ** (2/3)))), 1)
            else:
                Bval = max(int((noise * Ascale) // (3.61 * (gen_param ** (2/3)))), 1)
        else:
            Bval = max(Bscale, 1)

        for _ in range(num_trials):
            coefs = generator(p, gen_param, rng)
            lll_mat = recover_sparse_cyclotomic(coefs, p, Ascale=Ascale, Bscale=Bval, last_root=last_root, noise=noise, rng=rng, prec=prec)

            # Match successful candidate vectors
            for row in lll_mat:
                if abs(row[-1]) == 1:
                    # Construct desired histogram of roots of unity
                    expected_arr = [Bval * coef for coef in coefs]

                    # Check if candidate vector and desired histogram differ by all 1's vector
                    candidate = np.array(row)
                    if not last_root:
                        candidate[-1] = 0
                    diff_vec = row[-1] * candidate[2:p+2] - expected_arr
                    results[i] += int((np.abs(diff_vec - diff_vec[0]) < 10 ** -3).all())
                    break
        gc.collect()

    return results


def run_trials_timed(
        orders: list[int], 
        gen: str,
        gen_param: int, 
        Ascale: int, 
        Bscale: int, 
        last_root: bool, 
        num_trials: int, 
        noise: gmpy2.mpfr, 
        seed=None,
        prec=53
    ) -> tuple[list[int], list[float]]:
    """
    Tests success rate and times of using LLL algorithm to solve
    the cyclotomic integer decomposition integer relation problem.
    See `run_trials` for input parameter details.

    ### Returns:
    - `results`: list of number of successes corresponding to `orders`.
    - `times`: list of times (in seconds) needed to run all trials for each order in `orders`.
    """

    # Generate placeholder to record correct trials and time taken for each order
    results = [0] * len(orders)
    times = [0.0] * len(orders)

    # Initialize and set random cyclotomic integer generator
    rng = np.random.default_rng(seed)

    if gen == "l1pos":
        generator = coefficients_gen_l1_pos
    elif gen == "l1":
        generator = coefficients_gen_l1
    elif gen == "normal":
        generator = coefficients_gen_norm
    else:
        raise ValueError("Select cyclotomic integer generation mode `l1pos`, `l1`, or `normal`.")
    
    # Run trials for each order in orders
    for i in range(len(orders)):
        start_time = time.time()
        p = orders[i]

        # autocomputes Bscale if given noise
        if Bscale == 0 and noise != 0:
            if gen == "l1pos" or gen == "l1":
                Bval = max(int((noise * Ascale * p) // (4.5 * (gen_param ** (2/3)))), 1)
            else:
                Bval = max(int((noise * Ascale) // (3.61 * (gen_param ** (2/3)))), 1)
        else:
            Bval = max(Bscale, 1)

        for _ in range(num_trials):
            coefs = generator(p, gen_param, rng)
            lll_mat = recover_sparse_cyclotomic(coefs, p, Ascale=Ascale, Bscale=Bval, last_root=last_root, noise=noise, rng=rng, prec=prec)

            # Match successful candidate vectors
            for row in lll_mat:
                if abs(row[-1]) == 1:
                    # Construct desired histogram of roots of unity
                    expected_arr = [Bval * coef for coef in coefs]

                    # Check if candidate vector and desired histogram differ by all 1's vector
                    candidate = np.array(row)
                    if not last_root:
                        candidate[-1] = 0
                    diff_vec = row[-1] * candidate[2:p+2] - expected_arr
                    results[i] += int((np.abs(diff_vec - diff_vec[0]) < 10 ** -3).all())
                    break
        
        # Record times for order
        times[i] = time.time() - start_time

        gc.collect()
    
    return results, times