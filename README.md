![CI](https://github.com/DescriptorZoo/CHSF.jl/workflows/CI/badge.svg)  [![DOI](https://zenodo.org/badge/255959245.svg)](https://zenodo.org/badge/latestdoi/255959245)

# CHSF.jl
Julia code to generate Chebyshev Polynomial variant of Atom Centred Symmetry Functions (CHSF)

Chebyshev Polynomials Variant of Atom Centred Symmetry Functions
as they are defined in **N. Artrith and A. Urban, Comput. Mater. Sci. 114 (2016) 135-150.**

## Installation:

```
] add https://github.com/DescriptorZoo/CHSF.jl.git
] test CHSF
```

### Usage Example:

```
chsf(atoms, rcut, n=2, l=2)
```

### How to cite:

If you use this code, we would appreciate if you cite the following papers:
- Berk Onat, Christoph Ortner, James R. Kermode, 	J. Chem. Phys. 153, 144106 (2020) [Paper DOI:10.1063/5.0016005](https://doi.org/10.1063/5.0016005) [arXiv:2006.01915](https://arxiv.org/abs/2006.01915)
- Nongnuch Artrith, Alexander Urban, and Gerbrand Ceder, [Phys. Rev. B 96, 014112](https://doi.org/10.1103/PhysRevB.96.014112)
