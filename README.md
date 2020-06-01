![CI](https://github.com/DescriptorZoo/CHSF.jl/workflows/CI/badge.svg)  [![DOI](https://zenodo.org/badge/255959245.svg)](https://zenodo.org/badge/latestdoi/255959245)

# CHSF.jl
Julia port of Chebyshev Polynomial variant of Atom Centred Symmetry Functions (CHSF)

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
