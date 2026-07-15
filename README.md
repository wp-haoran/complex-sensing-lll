# Cyclotomic Integer Approximation

This repository contains the code accompanying:

> William H. Pan, Dylan Roscow, and Netanel Raviv. **"Complex Sensing: Reconstructing Dense Rational Signals from Few Complex Measurements."** *IEEE Signal Processing Letters* (in press).

## Overview

Our proposed complex sensing method reconstructs a dense (not necessarily sparse) rational signal from a small number of complex-valued linear measurements, using sensing matrices built from roots of unity. Recovering the signal from these measurements reduces to a **cyclotomic integer decoding** problem:

Given an $n^\text{th}$ root of unity $\zeta_n = e^{2\pi i / n}$ and a complex number

$$z = c_0 \zeta_n^0 + c_1 \zeta_n^1 + \cdots + c_{n-1} \zeta_n^{n-1},$$

recover the unknown integer coefficients $c_0, \dots, c_{n-1}$ from $z$ alone.

This is a shortest-vector-type problem with no known closed-form solution. This repository reduces cyclotomic integer decoding to finding a short vector in an explicitly constructed lattice, and solves it with the LLL algorithm [Lenstra, Lenstra, Lovász 1982]. We measure recovery success rate and runtime across root-of-unity order $p$, coefficient magnitude, numerical precision, and additive noise.

## Repository Contents

| File | Description |
|---|---|
| `lll.py` | Core library: random cyclotomic integer generators, LLL-based recovery (`recover_sparse_cyclotomic`), and batch trial runners (`run_trials`, `run_trials_timed`). |
| `examples.ipynb` | Reproduces the paper's experimental figures (success rate vs. precision/coefficient magnitude/noise) and illustrates runtime scaling using `lll.py`. |
| `testing.xlsx` | Raw trial data referenced by the plotting cells in `examples.ipynb`. |

## Installation

`fpylll` (the LLL implementation used here) is a standalone Python wrapper around the `fplll` C++ library. SageMath bundles it as part of its own environment. Otherwise, two install paths:

**Conda (recommended):**
```bash
conda install -c conda-forge fpylll
pip install -r requirements.txt
```
This pulls prebuilt binaries for `fplll`, `GMP`, and `MPFR`, so no local compilation is needed.

**Pip only:**
```bash
pip install -r requirements.txt
```
This will attempt to build `fpylll`'s C extension against your system's `fplll`/`GMP`/`MPFR` libraries, so install those first, e.g. on Debian/Ubuntu:
```bash
sudo apt install libfplll-dev libgmp-dev libmpfr-dev
```
(or the Homebrew equivalents on macOS). If you hit build errors, the conda route above is more reliable.

## Usage

The main entry point is `run_trials` in `lll.py`:

```python
from lll import run_trials

results = run_trials(
    orders=[5, 7, 11, 13, 17],   # root-of-unity orders p to test
    gen="l1",                     # coefficient distribution: "l1", "l1pos", or "normal"
    gen_param=2**20,              # distribution parameter (l1 radius or std. dev.)
    Ascale=10**16,                # lattice scaling factor for the objective value
    Bscale=5,                     # lattice scaling factor for the coefficient block
    last_root=False,              # include the p-1'th root of unity in the sum
    num_trials=1000,
    prec=53,                      # 53 = IEEE-754 double; larger = arbitrary precision via gmpy2
)
```

Key parameters:
- `Ascale` / `Bscale`: hyperparameters of the constructed lattice basis. In the noiseless case, best results come from setting `Ascale` as large as `prec` allows and `Bscale` around 3–5. In the noisy case, set `Bscale=0` to use the closed-form estimate derived in the paper.
- `prec`: number of mantissa bits. `prec=53` uses `numpy.float64` for speed; any other value switches to arbitrary-precision arithmetic via `gmpy2.mpfr`.

See `examples.ipynb` for full worked examples, and `testing.xlsx` for the parameter sweeps behind each figure in the paper.

## Dependencies

- [NumPy](https://numpy.org/)
- [gmpy2](https://gmpy2.readthedocs.io/en/latest/) — arbitrary-precision arithmetic
- [fpylll](https://github.com/fplll/fpylll) — LLL lattice reduction
- matplotlib

## Citation

```bibtex
@article{pan2025complex,
  title   = {Complex Sensing: Reconstructing Dense Rational Signals from Few Complex Measurements},
  author  = {Pan, William H. and Roscow, Dylan and Raviv, Netanel},
  journal = {IEEE Signal Processing Letters},
  year    = {2025},
  note    = {In press}
}
```

## Contact

For questions regarding the code, reach out to whpan [at] utexas [dot] edu or droscow [at] purdue [dot] edu.
