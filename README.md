# Experiment 5: Neural Network Layers in CUDA

## 1. Objective
To implement and evaluate neural network operations using CUDA and compare CPU and GPU execution performance.

## 2. Hardware and Software
- GPU: NVIDIA GeForce RTX 5060 Ti (16 GB)
- Operating System: Ubuntu 26.04 LTS (WSL2)
- CUDA Toolkit: 13.3
- Compiler: NVIDIA CUDA Compiler (nvcc)

## 3. Neural Network Operations
The experiment implements:
- Matrix multiplication with bias
- ReLU activation function
- Sigmoid activation function

## 4. Project Structure
```text
Experiment5_CUDA/
├── README.md
├── .gitignore
├── src/
│   └── neural_network.cu
├── results/
│   └── performance_results.txt
└── screenshots/

| Metric | Run 1 | Run 2 | Run 3 | Run 4 |
|---|---:|---:|---:|---:|
| CPU Average Time (ms) | 0.227644 | 0.223929 | 0.223802 | 0.223259 |
| GPU Kernel Time (ms) | 0.050862 | 0.059544 | 0.038038 | 0.028346 |
| GPU End-to-End Time (ms) | 0.263647 | 0.260239 | 0.226327 | 0.297853 |
| Kernel Speedup | 4.475683x | 3.760723x | 5.883593x | 7.876327x |
| End-to-End Speedup | 0.863441x | 0.860473x | 0.988845x | 0.749561x |

