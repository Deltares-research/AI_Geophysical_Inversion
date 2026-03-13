# AI Reconstruction of Geophysical Profile Using FNO

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

## Project Objective: Railway Instability Monitoring via Dark Fiber

### Problem Statement

This project uses FNO to **predict 2D S-wave velocity fields from seismic shotgathers** as a fast surrogate model for surface wave inversion. The key application is:

**Monitoring railway infrastructure instability using Dark Fiber DAS (Distributed Acoustic Sensing)**

### Why This Matters

- **Dark Fiber**: Unused optical fiber cables along railway networks can be repurposed as distributed sensors via Distributed Acoustic Sensing (DAS)
- **Early Warning**: Continuous monitoring of subsurface properties (S-wave velocity) helps detect structural degradation early
- **Cost-Effective**: Leverages existing fiber infrastructure instead of deploying traditional seismic sensors
- **Real-time Capability**: Traditional surface wave inversion is computationally expensive; FNO enables near real-time velocity model inference

### Traditional vs. FNO Approach

**Traditional Surface Wave Inversion:**
- Time-consuming (hours to days per solution)
- Requires iterative optimization loops
- Limited practical use for continuous monitoring

**FNO-Based Approach:**
- Forward prediction in seconds
- Pre-trained once, then deployed for rapid inference
- Enables continuous, real-time monitoring of railway stability

### Data Pipeline

```
Seismic Shotgathers (DAS recordings) 
    ↓
Input to FNO Model
    ↓
2D S-wave Velocity Field (prediction)
    ↓
Subsurface Property Assessment
    ↓
Railways Structural Health Monitoring
```

## Test Cases: Synthetic Data Validation

To validate the FNO concept, this project tests the model on **three increasingly complex synthetic geological scenarios**:

### Case I: Simple Two-Layer Model
- **Description**: Simplest scenario with two distinct horizontal layers
- **Purpose**: Baseline validation; tests if FNO can distinguish between simple layered structures
- **Velocity Profile**: Two constant layers with a clear boundary
- **Notebook**: `notebooks/CASE_I/Inference_CASE_I_source_31m.ipynb`

### Case II: Constant Medium with Embedded Anomaly
- **Description**: Homogeneous background with localized velocity anomalies
- **Purpose**: Tests FNO's ability to detect and localize isolated subsurface features
- **Features**: Point anomalies or small-scale heterogeneities embedded in constant velocity medium
- **Notebooks**: 
  - `notebooks/CASE_II/Inference_CASE_II_source_5m.ipynb`
  - `notebooks/CASE_II/Inference_CASE_II_source_31m.ipynb`
  - `notebooks/CASE_II/Inference_CASE_II_source_56m.ipynb`
  - `notebooks/CASE_II/Inference_CASE_II_Combined.ipynb`

### Case III: Complex Three-Layer Model (Stiff-Soft-Stiff)
- **Description**: Realistic geological scenario with variable velocity and wavy interfaces
- **Velocity Profile**: Three-layer structure with alternating stiffness (high-low-high S-wave velocity)
- **Complexity**: Non-planar interfaces, laterally varying velocity within layers
- **Purpose**: Tests FNO performance on realistic, complex subsurface models
- **Notebooks**:
  - `notebooks/COMPLEX/Inference_COMPLEX_6m_3Layers.ipynb`
  - `notebooks/COMPLEX/Inference_COMPLEX_31m.ipynb`
  - `notebooks/COMPLEX/Inference_COMPLEX_31m_v2.ipynb`

### Progression and Learning

The three-case progression tests:
1. **Simple structures** → Baseline accuracy on clean, layered models
2. **Anomaly detection** → Capability to identify localized features
3. **Complex geology** → Real-world applicability with variable velocity and irregular interfaces

Success across all three cases demonstrates FNO's robustness for railway monitoring applications where subsurface conditions vary from simple to complex.

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
