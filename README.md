# twin-prime-spectra

Builds two sparse bipartite graphs over the integers, one from prime sums and one from twin-prime sums, and compares the top singular values of their biadjacency matrices.

## Overview

This is an exploratory spectral-number-theory experiment. It constructs two bipartite graphs on integer index sets and asks whether their largest singular values differ in an interesting way:

- **Prime-sum graph**: an edge connects an even number and an odd number when their sum is prime.
- **Twin-prime graph**: an edge connects an even number and an odd number when their sum is prime and at least one of the sum +/- 2 is also prime (a twin-prime neighbor).

For each graph it computes the top-k singular values of the biadjacency matrix with a sparse SVD and plots the two spectra side by side. There are no claims here beyond the observed spectra. It is a visualization of how the twin-prime constraint reshapes the leading spectrum relative to the full prime-sum graph.

## How it works

1. **Sieve.** `sieve_primes` runs a sieve of Eratosthenes over `[0, max_sum]` and returns a boolean primality array. `max_sum` is set large enough to cover every possible even+odd sum (plus the +/- 2 offsets used by the twin-prime test).

2. **Bipartite graph construction.** Rows index even numbers `2*(i+1)`; columns index odd numbers `2*j+1`. For each even number, all candidate sums are computed at once as a vectorized array, and a boolean primality mask selects the edges. `build_biadjacency` keeps edges where the sum is prime; `build_twin_biadjacency` keeps edges where the sum is prime and `sum+2` or `sum-2` is also prime. Each graph is stored as a `scipy.sparse` CSR biadjacency matrix of shape `(n, n)`.

3. **Sparse SVD.** `top_k_eigenvalues` calls `scipy.sparse.linalg.svds` with `which='LM'` to get the k largest singular values, sorted descending. The singular values of a biadjacency matrix B are the nonnegative eigenvalues of the full bipartite adjacency `[[0, B], [B.T, 0]]`, so the singular spectrum is the natural object to compare.

4. **Spectral comparison.** `plot_spectrum` plots the top-k singular values of both graphs against rank on a shared axis (`matplotlib`), so the leading spectra can be compared directly. The values are also printed to stdout.

## Tech stack

- numpy
- scipy.sparse and scipy.sparse.linalg (CSR matrices, `svds`)
- matplotlib (spectrum plot)

## Setup

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# Unix:    source .venv/bin/activate
pip install -r requirements.txt
```

## Usage

The script takes two arguments via argparse: `-n` (matrix dimension, default 20000) and `-k` (number of top singular values, default 2).

```bash
# defaults: n = 20000, k = 2
python TwinPrimes.py

# explicit example
python TwinPrimes.py -n 10000 -k 5
```

It prints the top singular values for both graphs and then opens a matplotlib window with the spectral comparison plot. Larger `n` increases sieve and SVD cost; start small if you want a quick run.

## Results / notes

This is an exploratory experiment, not a result with a claim attached. Running it produces, for the chosen `n` and `k`:

- the top-k singular values of the prime-sum biadjacency matrix,
- the top-k singular values of the twin-prime biadjacency matrix,
- a comparison plot of the two spectra by rank.

The output is the observed spectra themselves. No conjecture, bound, or statistical claim about primes or twin primes is asserted here.

## Limitations

- Niche and exploratory. This is a curiosity-driven spectral probe, not a proof or a measured hypothesis test.
- Results depend on the chosen `n` and `k`, and on the specific (and somewhat arbitrary) bipartite encoding of even/odd index sets.
- No claims beyond the printed spectra and the comparison plot.
