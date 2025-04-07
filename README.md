# GPU-Accelerated American Option Pricing

## 📄 Papers Overview
This repository contains the implementation of GPU-accelerated American option pricing models based on two research papers:

1.	**Using the Graphics Processing Unit to Evaluate American-Style Derivatives**
      - Published in The Journal of Financial Data Science (JFDS)
      - Authors: Leon Xing Li, Ren-Raw Chen
      - This paper explores the use of GPU computing for American option pricing, leveraging Monte Carlo simulations (MCS) and Particle Swarm Optimization (PSO) to efficiently solve free-boundary PDEs. The approach achieves significant performance gains over CPU-based methods, making it ideal for exotic derivatives and large financial portfolios.
2.	**GPU-Accelerated American Option Pricing: The Case of the Longstaff-Schwartz Monte Carlo Model**
      - Published in The Journal of Derivatives (JOD)
      - Authors: Leon Xing Li, Ren-Raw Chen, Frank J. Fabozzi
      - This paper implements the Longstaff-Schwartz Monte Carlo (LSMC) model on GPUs, addressing computational challenges such as instability in basis function selection. It optimizes regression calculations for high-performance Single Instruction Multiple Data (SIMD) processing, enabling fast and accurate American option pricing.

The repository provides code implementations for evaluating American-style derivatives using GPU computing, with a focus on Monte Carlo simulation (MCS), Particle Swarm Optimization (PSO), and the Longstaff-Schwartz Monte Carlo (LSMC) method.

## 🚀 Features
- Parallelized Monte Carlo Simulations: Efficiently handles large-scale option pricing using GPU acceleration.
- Particle Swarm Optimization (PSO): Computes the early exercise boundary for American options.
- LSMC Model with Stability Enhancements: Implements regression adjustments to prevent numerical instability.
- Cross-Platform Compatibility: Built with OpenCL for portability across different GPU architectures (NVIDIA, AMD, Apple M Series).
- Python + PyOpenCL: Integrates GPU computation within Python for easy experimentation.

## 📊 Key Contributions
- **Monte Carlo Simulation & PSO on GPU:** Utilization of PyOpenCL to accelerate American option pricing via Monte Carlo simulations and PSO.
- **Longstaff-Schwartz Monte Carlo on GPU:** Implementation of the LSMC approach using OpenCL, with optimizations for numerical stability and parallel efficiency.
- **Performance Benchmarks:** Comparison between CPU and GPU implementations, demonstrating significant speed improvements using GPU computing.
- **Hardware-Agnostic Design:** Adoption of OpenCL for cross-platform GPU acceleration, supporting both Nvidia and non-Nvidia GPUs.

## ⚙️ Implementation Details
### 1. American Option Pricing using Monte Carlo Simulation & PSO
- **Objective:** Solve the free-boundary PDE problem for American options.
- **Methodology:**
  - Monte Carlo simulation for stock price paths.
  - Particle Swarm Optimization (PSO) to identify early exercise boundaries.
  - OpenCL-based parallelization for GPU acceleration.
- **Performance:** Achieves up to 300x speed improvement over CPU-based approaches.

### 2. GPU-Accelerated Longstaff-Schwartz Monte Carlo (LSMC) Method
- **Objective:** Enhance LSMC efficiency using GPU computing.
- **Methodology:**
  - Regression-based continuation value estimation.
  - Basis function optimization for numerical stability.
  - Matrix operations (transpose, inversion, multiplication) implemented in OpenCL.
- **Performance:** GPU-optimized regression computations reduce instability while maintaining speed improvements.

## 📂 Repository Structure
📁 gpu-american-options  
│── 📜 README.md  
│── 📂 src/  
│   ├── American_option.py                                        # CLI option pricing implementation  
│   ├── American_option.ipynb                                     # Jupyter Notebook option pricing implementation 
│   ├── 📂 models/     
│       ├── benchmarks.py                                         # Benchmark functions  
│       ├── longstaff.py                                          # Longstaff Schwartz functions  
│       ├── mc.py                                                 # Monte Carlo simulation functions  
│       ├── pso.py                                                # PSO functions  
│       ├── utils.py                                              # OpenCL Helper functions  
│       ├── 📂 kernels/                       
│           ├── knl_source_pso_computeCosts.c                     # Kernel for PSO computing costs to optimize  
│           ├── knl_source_pso_searchGrid.c                       # Kernel for PSO search grid
│           ├── knl_src_pre_calc_ClassicAdjoint.c                 # Kernel for classic adjoint matrix inversion
│           ├── knl_src_pre_calc_GaussJordan.c                    # Kernel for Gauss-Jordan Elimination matrix inversion  
│
│── 📂 design/  
│   ├── Longstaff_Schwartz.xlsx        # LSMC analysis 
│   ├── LSMC_memory-data-layout.xlsx   # LSMC GPU algo memory layout design
│   ├── analysis.xlsx                  # PSO implementation  
│   ├── psoOption_design.xlsx          # PSO GPU algo memory layout design

## 🛠 Installation
To run the implementations, install the required dependencies:<p>
```pip install numpy scipy pyopencl matplotlib```

## 🧩 Usage
Run the model for American option pricing:<p>
```python3 src/American_option.py```

## 📜 Citations
If you use this code, please cite the following papers:

**Li, L. X., & Chen, R. R. (2023).** Using the Graphics Processing Unit to Evaluate American-Style Derivatives. *Journal of Financial Data Science.*

**Li, L. X., Chen, R. R., & Fabozzi, F. J. (2024).** GPU-Accelerated American Option Pricing: The Case of the Longstaff-Schwartz Monte Carlo Model. *Journal of Derivatives.*

## 📧 Contact
For questions or collaborations, please contact **Leon Xing Li** at [leonchao@yeah.net](mailto:leonchao@yeah.net).

## 📝 License
MIT License - See LICENSE for details.

