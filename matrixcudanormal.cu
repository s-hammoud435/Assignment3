#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <time.h>

#define BLOCK_SIZE 16

__global__ void multiplyMatrices(double *matrix1, double *matrix2, double *resultMatrix, int rows1, int commonDim, int cols2) {
    int rowIndex = blockIdx.y * blockDim.y + threadIdx.y;
    int colIndex = blockIdx.x * blockDim.x + threadIdx.x;

    if (rowIndex < rows1 && colIndex < cols2) {
        double sum = 0.0;
        for (int i = 0; i < commonDim; i++) {
            sum += matrix1[rowIndex * commonDim + i] * matrix2[i * cols2 + colIndex];
        }
        resultMatrix[rowIndex * cols2 + colIndex] = sum;
    }
}

double* generateRandomMatrix(int elements) {
    double* matrix = (double*)malloc(elements * sizeof(double));
    for (int i = 0; i < elements; i++) {
        matrix[i] = (double)rand() / RAND_MAX * 100;
    }
    return matrix;
}

int main() {
    srand(time(NULL));
    int rowsA = 500, colsA = 300, colsB = 400;

    size_t sizeMatrixA = rowsA * colsA * sizeof(double);
    size_t sizeMatrixB = colsA * colsB * sizeof(double);
    size_t sizeMatrixC = rowsA * colsB * sizeof(double);

    double *hostMatrix1 = generateRandomMatrix(rowsA * colsA);
    double *hostMatrix2 = generateRandomMatrix(colsA * colsB);
    double *hostResult = (double*)malloc(sizeMatrixC);

    double *deviceMatrix1, *deviceMatrix2, *deviceResult;
    cudaMalloc(&deviceMatrix1, sizeMatrixA);
    cudaMalloc(&deviceMatrix2, sizeMatrixB);
    cudaMalloc(&deviceResult, sizeMatrixC);

    cudaMemcpy(deviceMatrix1, hostMatrix1, sizeMatrixA, cudaMemcpyHostToDevice);
    cudaMemcpy(deviceMatrix2, hostMatrix2, sizeMatrixB, cudaMemcpyHostToDevice);

    dim3 blockDims(BLOCK_SIZE, BLOCK_SIZE);
    dim3 gridDims((colsB + BLOCK_SIZE- 1) / BLOCK_SIZE, (rowsA + BLOCK_SIZE- 1) / BLOCK_SIZE);

    cudaEvent_t startEvent, stopEvent;
    cudaEventCreate(&startEvent);
    cudaEventCreate(&stopEvent);

    float elapsedMilliseconds = 0;
    float totalElapsedSeconds = 0;

    for (int i = 0; i < 10; i++) {
        cudaEventRecord(startEvent);
        multiplyMatrices<<<gridDims, blockDims>>>(deviceMatrix1, deviceMatrix2, deviceResult, rowsA, colsA, colsB);
        cudaEventRecord(stopEvent);
        cudaEventSynchronize(stopEvent);

        cudaEventElapsedTime(&elapsedMilliseconds, startEvent, stopEvent);
        totalElapsedSeconds += elapsedMilliseconds / 1000.0;
    }

    printf("Average time taken over 10 iterations: %.6f seconds\n", totalElapsedSeconds / 10);

    cudaMemcpy(hostResult, deviceResult, sizeMatrixC, cudaMemcpyDeviceToHost);

    cudaFree(deviceMatrix1);
    cudaFree(deviceMatrix2);
    cudaFree(deviceResult);

    free(hostMatrix1);
    free(hostMatrix2);
    free(hostResult);

    cudaEventDestroy(startEvent);
    cudaEventDestroy(stopEvent);

    return 0;
}
