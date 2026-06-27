#!/usr/bin/env python3
"""
twin_prime_spectral.py

Efficiently compare the top-k eigenvalues (spectrum) of:
 - full prime-sum bipartite graph
 - twin-prime bipartite graph
for large n (e.g. n up to 10000).
Only plots the spectral comparison of the largest k eigenvalues.
"""
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import matplotlib.pyplot as plt
import argparse


def sieve_primes(limit):
    """Linear sieve returning boolean array up to limit."""
    is_prime = np.ones(limit+1, dtype=bool)
    is_prime[:2] = False
    for i in range(2, int(limit**0.5)+1):
        if is_prime[i]:
            is_prime[i*i : limit+1 : i] = False
    return is_prime


def build_biadjacency(n, is_prime):
    """Construct sparse biadjacency B for prime-sum graph."""
    evens = 2*(np.arange(n)+1)
    odds = 2*np.arange(n)+1
    rows, cols = [], []
    for i, e in enumerate(evens):
        # potential sums = e + odds
        sums = e + odds
        mask = is_prime[sums]
        js = np.nonzero(mask)[0]
        rows.extend([i]*len(js))
        cols.extend(js.tolist())
    data = np.ones(len(rows), dtype=int)
    return sp.csr_matrix((data, (rows, cols)), shape=(n, n))


def build_twin_biadjacency(n, is_prime):
    """Construct sparse biadjacency for twin-prime graph (numeric): sum and sum±2 primes."""
    evens = 2*(np.arange(n)+1)
    odds = 2*np.arange(n)+1
    rows, cols = [], []
    for i, e in enumerate(evens):
        sums = e + odds
        mask = is_prime[sums] & ((is_prime[sums+2]) | (is_prime[sums-2]))
        js = np.nonzero(mask)[0]
        rows.extend([i]*len(js))
        cols.extend(js.tolist())
    data = np.ones(len(rows), dtype=int)
    return sp.csr_matrix((data, (rows, cols)), shape=(n, n))


def top_k_eigenvalues(B, k):
    """Compute top-k singular values of B (eigenvalues of bipartite adjacency)."""
    # singular values of B correspond to positive eigenvalues of A = [[0,B],[B.T,0]]
    u, s, vt = spla.svds(B, k=k, which='LM')
    s = np.sort(s)[::-1]
    # bipartite adjacency eigenvalues: ±s
    return s


def plot_spectrum(s_full, s_twin, k):
    idx = np.arange(1, k+1)
    plt.figure(figsize=(6,4))
    plt.plot(idx, s_full, 'o-', label='Full prime-sum')
    plt.plot(idx, s_twin, 'o-', label='Twin-prime only')
    plt.title(f'Top-{k} Singular Values Comparison')
    plt.xlabel('Rank')
    plt.ylabel('Singular Value')
    plt.legend()
    plt.tight_layout()
    plt.show()


def main():
    parser = argparse.ArgumentParser(description='Spectral comparison for large n')
    parser.add_argument('-n', type=int, default=20000, help='matrix dimension')
    parser.add_argument('-k', type=int, default=2, help='number of top eigenvalues')
    args = parser.parse_args()

    n, k = args.n, args.k
    max_sum = 2*(n+1) + (2*(n-1)+1) + 2
    is_prime = sieve_primes(max_sum)

    print(f'[+] Building prime-sum biadjacency for n={n}...')
    B = build_biadjacency(n, is_prime)
    print(f'[+] Building twin-prime biadjacency for n={n}...')
    T = build_twin_biadjacency(n, is_prime)

    print(f'[+] Computing top-{k} singular values...')
    s_full = top_k_eigenvalues(B, k)
    s_twin = top_k_eigenvalues(T, k)

    print('Full top values:', np.round(s_full,4))
    print('Twin top values:', np.round(s_twin,4))

    plot_spectrum(s_full, s_twin, k)

if __name__ == '__main__':
    main()
