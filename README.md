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

If you use this code, please cite the following papers:
- Nongnuch Artrith and Alexander Urban, [Comput. Mater. Sci. 114 (2016) 135-150.](https://doi.org/10.1016/j.commatsci.2015.11.047)
- Berk Onat, Christoph Ortner, James R. Kermode, 	[arXiv:2006.01915 (2020)](https://arxiv.org/abs/2006.01915)
