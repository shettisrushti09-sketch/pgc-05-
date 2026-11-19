
// Experiment 5: Neural Network Layers in CUDA
// Implements matrix multiplication + bias, ReLU, and sigmoid.
// Compares a CPU forward pass with a custom CUDA forward pass.

#include <cuda_runtime.h>
#include <iostream>
#include <vector>
#include <cmath>
#include <chrono>
#include <algorithm>
#include <iomanip>
#include <cstdlib>

#define CUDA_CHECK(call) do { \
    cudaError_t err = (call); \
    if (err != cudaSuccess) { \
        std::cerr << "CUDA error at " << __FILE__ << ":" << __LINE__ \
                  << " - " << cudaGetErrorString(err) << std::endl; \
        std::exit(EXIT_FAILURE); \
    } \
} while (0)

constexpr int BATCH = 256;
constexpr int INPUT_SIZE = 128;
constexpr int HIDDEN_SIZE = 64;
constexpr int OUTPUT_SIZE = 10;
constexpr int REPEATS = 20;


// ============================================================
// PHASE 2: CUDA KERNELS
// ============================================================

// Matrix multiplication with bias
__global__ void matrixMulBiasKernel(
    const float* A, const float* B,
    const float* bias, float* C,
    int M, int N, int K)
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < M && col < N) {
        float sum = 0.0f;

        for (int k = 0; k < K; ++k) {
            sum += A[row * K + k] * B[k * N + col];
        }

        C[row * N + col] = sum + bias[col];
    }
}

// ReLU activation
__global__ void reluKernel(float* values, int count)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < count) {
        values[idx] = fmaxf(0.0f, values[idx]);
    }
}

// Sigmoid activation
__global__ void sigmoidKernel(float* values, int count)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < count) {
        values[idx] = 1.0f / (1.0f + expf(-values[idx]));
    }
}


// ============================================================
// PHASE 3: CPU IMPLEMENTATION
// ============================================================

void cpuMatMulBias(
    const std::vector<float>& A,
    const std::vector<float>& B,
    const std::vector<float>& bias,
    std::vector<float>& C,
    int M, int N, int K)
{
    for (int row = 0; row < M; ++row) {
        for (int col = 0; col < N; ++col) {
            float sum = 0.0f;

            for (int k = 0; k < K; ++k) {
                sum += A[row * K + k] * B[k * N + col];
            }

            C[row * N + col] = sum + bias[col];
        }
    }
}

void cpuReLU(std::vector<float>& values)
{
    for (float& x : values) {
        x = std::max(0.0f, x);
    }
}

void cpuSigmoid(std::vector<float>& values)
{
    for (float& x : values) {
        x = 1.0f / (1.0f + std::exp(-x));
    }
}

void cpuForwardPass(
    const std::vector<float>& X,
    const std::vector<float>& W1,
    const std::vector<float>& b1,
    const std::vector<float>& W2,
    const std::vector<float>& b2,
    std::vector<float>& output)
{
    std::vector<float> z1(BATCH * HIDDEN_SIZE);

    cpuMatMulBias(
        X, W1, b1, z1,
        BATCH, HIDDEN_SIZE, INPUT_SIZE
    );

    cpuReLU(z1);

    std::vector<float> z2(BATCH * OUTPUT_SIZE);

    cpuMatMulBias(
        z1, W2, b2, z2,
        BATCH, OUTPUT_SIZE, HIDDEN_SIZE
    );

    cpuSigmoid(z2);
    output = z2;
}


// ============================================================
// PHASE 4: GPU FORWARD PASS
// ============================================================

void launchGpuForwardPass(
    const float* dX,
    const float* dW1,
    const float* db1,
    float* dZ1,
    const float* dW2,
    const float* db2,
    float* dZ2)
{
    dim3 block(16, 16);

    dim3 grid1(
        (HIDDEN_SIZE + block.x - 1) / block.x,
        (BATCH + block.y - 1) / block.y
    );

    matrixMulBiasKernel<<<grid1, block>>>(
        dX, dW1, db1, dZ1,
        BATCH, HIDDEN_SIZE, INPUT_SIZE
    );

    int hiddenCount = BATCH * HIDDEN_SIZE;
    int threads = 256;
    int blocks = (hiddenCount + threads - 1) / threads;

    reluKernel<<<blocks, threads>>>(dZ1, hiddenCount);

    dim3 grid2(
        (OUTPUT_SIZE + block.x - 1) / block.x,
        (BATCH + block.y - 1) / block.y
    );

    matrixMulBiasKernel<<<grid2, block>>>(
        dZ1, dW2, db2, dZ2,
        BATCH, OUTPUT_SIZE, HIDDEN_SIZE
    );

    int outputCount = BATCH * OUTPUT_SIZE;
    int outputBlocks = (outputCount + threads - 1) / threads;

    sigmoidKernel<<<outputBlocks, threads>>>(dZ2, outputCount);

    CUDA_CHECK(cudaGetLastError());
}

bool nearlyEqual(
    float a,
    float b,
    float tolerance = 1e-5f)
{
    return std::fabs(a - b) <= tolerance;
}


// ============================================================
// PHASE 5: INDIVIDUAL CUDA KERNEL TESTS
// ============================================================

bool testMatrixKernel()
{
    const int M = 2, N = 2, K = 2;

    std::vector<float> A = {1, 2, 3, 4};
    std::vector<float> B = {5, 6, 7, 8};
    std::vector<float> bias = {1, 1};
    std::vector<float> C(M * N);

    float *dA, *dB, *dBias, *dC;

    CUDA_CHECK(cudaMalloc(&dA, A.size() * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&dB, B.size() * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&dBias, bias.size() * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&dC, C.size() * sizeof(float)));

    CUDA_CHECK(cudaMemcpy(
        dA, A.data(), A.size() * sizeof(float),
        cudaMemcpyHostToDevice
    ));

    CUDA_CHECK(cudaMemcpy(
        dB, B.data(), B.size() * sizeof(float),
        cudaMemcpyHostToDevice
    ));

    CUDA_CHECK(cudaMemcpy(
        dBias, bias.data(), bias.size() * sizeof(float),
        cudaMemcpyHostToDevice
    ));

    dim3 block(16, 16);
    dim3 grid((N + 15) / 16, (M + 15) / 16);

    matrixMulBiasKernel<<<grid, block>>>(
        dA, dB, dBias, dC, M, N, K
    );

    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaMemcpy(
        C.data(), dC, C.size() * sizeof(float),
        cudaMemcpyDeviceToHost
    ));

    bool passed =
        nearlyEqual(C[0], 20.0f) &&
        nearlyEqual(C[1], 23.0f) &&
        nearlyEqual(C[2], 44.0f) &&
        nearlyEqual(C[3], 51.0f);

    cudaFree(dA);
    cudaFree(dB);
    cudaFree(dBias);
    cudaFree(dC);

    return passed;
}


bool testReLUKernel()
{
    std::vector<float> values = {
        -2.0f, -0.5f, 0.0f, 1.5f, 3.0f
    };

    float* dValues;

    CUDA_CHECK(cudaMalloc(
        &dValues, values.size() * sizeof(float)
    ));

    CUDA_CHECK(cudaMemcpy(
        dValues, values.data(),
        values.size() * sizeof(float),
        cudaMemcpyHostToDevice
    ));

    reluKernel<<<1, 256>>>(
        dValues, static_cast<int>(values.size())
    );

    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaMemcpy(
        values.data(), dValues,
        values.size() * sizeof(float),
        cudaMemcpyDeviceToHost
    ));

    bool passed =
        nearlyEqual(values[0], 0.0f) &&
        nearlyEqual(values[1], 0.0f) &&
        nearlyEqual(values[2], 0.0f) &&
        nearlyEqual(values[3], 1.5f) &&
        nearlyEqual(values[4], 3.0f);

    cudaFree(dValues);

    return passed;
}


bool testSigmoidKernel()
{
    std::vector<float> values = {
        -2.0f, 0.0f, 2.0f
    };

    float* dValues;

    CUDA_CHECK(cudaMalloc(
        &dValues, values.size() * sizeof(float)
    ));

    CUDA_CHECK(cudaMemcpy(
        dValues, values.data(),
        values.size() * sizeof(float),
        cudaMemcpyHostToDevice
    ));

    sigmoidKernel<<<1, 256>>>(
        dValues, static_cast<int>(values.size())
    );

    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaMemcpy(
        values.data(), dValues,
        values.size() * sizeof(float),
        cudaMemcpyDeviceToHost
    ));

    bool passed =
        nearlyEqual(values[0], 0.1192029f) &&
        nearlyEqual(values[1], 0.5f) &&
        nearlyEqual(values[2], 0.8807971f);

    cudaFree(dValues);

    return passed;
}


// ============================================================
// PHASE 6, 7 AND 8: MAIN PROGRAM
// ============================================================

int main()
{
    // Detect CUDA GPU
    int deviceCount = 0;

    CUDA_CHECK(cudaGetDeviceCount(&deviceCount));

    if (deviceCount == 0) {
        std::cerr << "No CUDA-capable NVIDIA GPU was detected."
                  << std::endl;
        return EXIT_FAILURE;
    }

    CUDA_CHECK(cudaSetDevice(0));

    cudaDeviceProp prop{};

    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));

    std::cout << "EXPERIMENT 5: NEURAL NETWORK LAYERS IN CUDA\n";
    std::cout << "GPU detected: " << prop.name << "\n\n";


    // Test individual CUDA kernels
    std::cout << "Testing individual CUDA kernels...\n";

    bool matrixOK = testMatrixKernel();
    bool reluOK = testReLUKernel();
    bool sigmoidOK = testSigmoidKernel();

    std::cout << "Matrix multiplication + bias: "
              << (matrixOK ? "PASSED" : "FAILED") << "\n";

    std::cout << "ReLU: "
              << (reluOK ? "PASSED" : "FAILED") << "\n";

    std::cout << "Sigmoid: "
              << (sigmoidOK ? "PASSED" : "FAILED") << "\n";

    if (!(matrixOK && reluOK && sigmoidOK)) {
        std::cerr << "A kernel test failed. Stop and debug before continuing.\n";
        return EXIT_FAILURE;
    }


    // Initialize input data and weights
    std::vector<float> X(BATCH * INPUT_SIZE);
    std::vector<float> W1(INPUT_SIZE * HIDDEN_SIZE);
    std::vector<float> b1(HIDDEN_SIZE);
    std::vector<float> W2(HIDDEN_SIZE * OUTPUT_SIZE);
    std::vector<float> b2(OUTPUT_SIZE);

    for (size_t i = 0; i < X.size(); ++i)
        X[i] = 0.5f * std::sin(static_cast<float>(i) * 0.013f);

    for (size_t i = 0; i < W1.size(); ++i)
        W1[i] = 0.08f * std::cos(static_cast<float>(i) * 0.017f);

    for (size_t i = 0; i < b1.size(); ++i)
        b1[i] = 0.01f * static_cast<float>(i % 7);

    for (size_t i = 0; i < W2.size(); ++i)
        W2[i] = 0.10f * std::sin(static_cast<float>(i) * 0.019f);

    for (size_t i = 0; i < b2.size(); ++i)
        b2[i] = 0.02f * static_cast<float>(i % 5);


    // CPU forward pass and timing
    std::vector<float> cpuOutput(BATCH * OUTPUT_SIZE);

    auto cpuStart = std::chrono::steady_clock::now();

    for (int r = 0; r < REPEATS; ++r) {
        cpuForwardPass(X, W1, b1, W2, b2, cpuOutput);
    }

    auto cpuStop = std::chrono::steady_clock::now();

    double cpuMs =
        std::chrono::duration<double, std::milli>(
            cpuStop - cpuStart
        ).count() / REPEATS;


    // Allocate GPU memory
    float *dX = nullptr, *dW1 = nullptr;
    float *db1 = nullptr, *dZ1 = nullptr;
    float *dW2 = nullptr, *db2 = nullptr;
    float *dZ2 = nullptr;

    CUDA_CHECK(cudaMalloc(&dX, X.size() * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&dW1, W1.size() * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&db1, b1.size() * sizeof(float)));
    CUDA_CHECK(cudaMalloc(
        &dZ1, BATCH * HIDDEN_SIZE * sizeof(float)
    ));
    CUDA_CHECK(cudaMalloc(&dW2, W2.size() * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&db2, b2.size() * sizeof(float)));
    CUDA_CHECK(cudaMalloc(
        &dZ2, BATCH * OUTPUT_SIZE * sizeof(float)
    ));


    // Copy data from CPU to GPU
    CUDA_CHECK(cudaMemcpy(
        dX, X.data(), X.size() * sizeof(float),
        cudaMemcpyHostToDevice
    ));

    CUDA_CHECK(cudaMemcpy(
        dW1, W1.data(), W1.size() * sizeof(float),
        cudaMemcpyHostToDevice
    ));

    CUDA_CHECK(cudaMemcpy(
        db1, b1.data(), b1.size() * sizeof(float),
        cudaMemcpyHostToDevice
    ));

    CUDA_CHECK(cudaMemcpy(
        dW2, W2.data(), W2.size() * sizeof(float),
        cudaMemcpyHostToDevice
    ));

    CUDA_CHECK(cudaMemcpy(
        db2, b2.data(), b2.size() * sizeof(float),
        cudaMemcpyHostToDevice
    ));


    // GPU warm-up
    for (int r = 0; r < 5; ++r) {
        launchGpuForwardPass(
            dX, dW1, db1, dZ1, dW2, db2, dZ2
        );
    }

    CUDA_CHECK(cudaDeviceSynchronize());


    // Measure GPU kernel execution time
    cudaEvent_t start, stop;

    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    CUDA_CHECK(cudaEventRecord(start));

    for (int r = 0; r < REPEATS; ++r) {
        launchGpuForwardPass(
            dX, dW1, db1, dZ1, dW2, db2, dZ2
        );
    }

    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));

    float kernelMs = 0.0f;

    CUDA_CHECK(cudaEventElapsedTime(
        &kernelMs, start, stop
    ));

    kernelMs /= REPEATS;

    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));


    // Measure end-to-end GPU execution time
    std::vector<float> gpuOutput(BATCH * OUTPUT_SIZE);

    auto totalStart = std::chrono::steady_clock::now();

    for (int r = 0; r < REPEATS; ++r) {

        CUDA_CHECK(cudaMemcpy(
            dX, X.data(), X.size() * sizeof(float),
            cudaMemcpyHostToDevice
        ));

        CUDA_CHECK(cudaMemcpy(
            dW1, W1.data(), W1.size() * sizeof(float),
            cudaMemcpyHostToDevice
        ));

        CUDA_CHECK(cudaMemcpy(
            db1, b1.data(), b1.size() * sizeof(float),
            cudaMemcpyHostToDevice
        ));

        CUDA_CHECK(cudaMemcpy(
            dW2, W2.data(), W2.size() * sizeof(float),
            cudaMemcpyHostToDevice
        ));

        CUDA_CHECK(cudaMemcpy(
            db2, b2.data(), b2.size() * sizeof(float),
            cudaMemcpyHostToDevice
        ));

        launchGpuForwardPass(
            dX, dW1, db1, dZ1, dW2, db2, dZ2
        );

        CUDA_CHECK(cudaDeviceSynchronize());

        CUDA_CHECK(cudaMemcpy(
            gpuOutput.data(), dZ2,
            gpuOutput.size() * sizeof(float),
            cudaMemcpyDeviceToHost
        ));
    }

    auto totalStop = std::chrono::steady_clock::now();

    double totalGpuMs =
        std::chrono::duration<double, std::milli>(
            totalStop - totalStart
        ).count() / REPEATS;


    // Compare CPU and GPU outputs
    float maxAbsError = 0.0f;
    bool outputsMatch = true;

    for (size_t i = 0; i < cpuOutput.size(); ++i) {

        float error = std::fabs(
            cpuOutput[i] - gpuOutput[i]
        );

        maxAbsError = std::max(maxAbsError, error);

        if (!nearlyEqual(
            cpuOutput[i], gpuOutput[i], 1e-4f
        )) {
            outputsMatch = false;
        }
    }


    // Calculate speedup
    double kernelSpeedup = cpuMs / kernelMs;
    double endToEndSpeedup = cpuMs / totalGpuMs;


    // Display results
    std::cout << std::fixed << std::setprecision(6);

    std::cout << "\n========== PERFORMANCE RESULTS ==========\n";

    std::cout << "CPU average time: "
              << cpuMs << " ms\n";

    std::cout << "GPU kernel average time: "
              << kernelMs << " ms\n";

    std::cout << "GPU end-to-end average time: "
              << totalGpuMs << " ms\n";

    std::cout << "Kernel speedup: "
              << kernelSpeedup << "x\n";

    std::cout << "End-to-end speedup: "
              << endToEndSpeedup << "x\n";

    std::cout << "Maximum absolute error: "
              << maxAbsError << "\n";

    std::cout << "CPU and GPU outputs: "
              << (outputsMatch ? "MATCH" : "DO NOT MATCH")
              << "\n";


    // Display first 10 output values
    std::cout << "\nFirst 10 output values:\n";

    for (int i = 0; i < 10; ++i) {
        std::cout << "CPU: " << cpuOutput[i]
                  << " | GPU: " << gpuOutput[i] << "\n";
    }


    // Release GPU memory
    CUDA_CHECK(cudaFree(dX));
    CUDA_CHECK(cudaFree(dW1));
    CUDA_CHECK(cudaFree(db1));
    CUDA_CHECK(cudaFree(dZ1));
    CUDA_CHECK(cudaFree(dW2));
    CUDA_CHECK(cudaFree(db2));
    CUDA_CHECK(cudaFree(dZ2));

    CUDA_CHECK(cudaDeviceReset());

    return outputsMatch ? EXIT_SUCCESS : EXIT_FAILURE;
}
