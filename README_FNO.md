# Fourier Neural Operator (FNO) - Basic Concepts

## Overview

**Fourier Neural Operator (FNO)** is a deep learning architecture designed to solve parametric partial differential equations (PDEs) and other physics-based problems. Unlike traditional neural networks that work in physical space, FNO operates in the Fourier (frequency) domain, making it particularly effective for modeling complex spatial relationships and wave-like phenomena.

## Key Concepts

### 1. **What is FNO?**

FNO is a type of neural operator that:
- Maps functions to functions (rather than vectors to vectors like standard neural networks)
- Learns to approximate the solution operator of PDEs
- Works by performing operations in Fourier space, then converting back to physical space
- Can generalize across different input sizes and resolutions

### 2. **Why Fourier Domain?**

The Fourier transform decomposes signals into their frequency components. Operating in this domain offers several advantages:
- **Efficiency**: Many physical phenomena are naturally sparse in frequency space
- **Global Information**: Fourier modes capture long-range dependencies efficiently
- **Parameterization**: Fewer parameters needed to represent complex spatial patterns
- **Speed**: Fast Fourier Transform (FFT) provides computational efficiency

### 3. **How FNO Works**

The basic operation flow is:

```
Input Data → Fourier Transform → Linear Transformation (in frequency domain) 
           → Inverse Fourier Transform → Output Data
```

Each FNO layer:
1. **Applies Fourier Transform** to the input
2. **Performs linear transformation** on selected frequency modes (truncated to manageable number)
3. **Applies Inverse Fourier Transform** to return to physical space
4. **Adds a local component** (typically a simple MLP) for non-linear behavior

### 4. **FNO Architecture in PhysicsNeMo**

This implementation includes:
- **fno_layers**: Multiple stacked FNO layers for feature learning
- **fno_modes**: Number of frequency modes to retain (controls complexity/efficiency trade-off)
- **padding**: Zero-padding applied before FFT to reduce aliasing effects
- **decoder**: Maps learned features to output variables

Example configuration:
```yaml
fno:
  nr_fno_layers: 4      # Number of FNO layers
  fno_modes: 12         # Number of Fourier modes
  dimension: 2          # 2D problem
  padding: 9            # Padding size
```

## Advantages of FNO

✓ **Faster than traditional solvers**: Can solve PDEs orders of magnitude faster than numerical methods  
✓ **Resolution-agnostic**: Trained on one grid resolution, can sometimes be applied to others  
✓ **Efficient learning**: Learns global dependencies effectively with fewer parameters  
✓ **Generalizable**: Works well for learning operators across different problem parameters  

## Applications

FNO is particularly useful for:
- Seismic wave propagation (like in this geophysical inversion project)
- Weather prediction
- Climate modeling
- Turbulence simulation
- Fluid dynamics
- Material property prediction

## Related Files

- **Config**: `scripts/config/config_FNO_ver02.yaml` - FNO architecture configuration
- **Utilities**: `scripts/ops.py` - Numerical operations (derivatives, convolutions)
- **Notebooks**: Various inference examples in `notebooks/` directory

## References

- **Original FNO Paper**: Li et al., "Fourier Neural Operator for Parametric Partial Differential Equations" (ICLR 2021)
- **PhysicsNeMo**: NVIDIA's framework for physics-informed neural operators
- **Link**: https://github.com/NVIDIA/modulus

---

**Note**: This project implements FNO for geophysical inversion tasks, specifically for modeling seismic wave propagation and inferring subsurface properties.
