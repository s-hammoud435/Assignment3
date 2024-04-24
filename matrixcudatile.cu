#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <time.h>

#define TILE_SIZE 16

__global__ void matrixMultiply(double *M1, double *M2, double *Result, int rowsA, int colsA, int colsB) {
    __shared__ double sharedM1[TILE_SIZE][TILE_SIZE];
    __shared__ double sharedM2[TILE_SIZE][TILE_SIZE];
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    double tempSum = 0.0;

    for (int tileIndex = 0; tileIndex < (colsA + TILE_SIZE - 1) / TILE_SIZE; tileIndex++) {
        int indexM1 = row * colsA + tileIndex * TILE_SIZE + threadIdx.x;
        if (indexM1 / colsA == row && indexM1 % colsA < colsA) {  
            sharedM1[threadIdx.y][threadIdx.x] = M1[indexM1];
        } else {
            sharedM1[threadIdx.y][threadIdx.x] = 0.0;
        }

        int indexM2 = (tileIndex * TILE_SIZE + threadIdx.y) * colsB + col;
        if (indexM2 / colsB == tileIndex * TILE_SIZE + threadIdx.y && indexM2 % colsB == col) {
            sharedM2[threadIdx.y][threadIdx.x] = M2[indexM2];
        } else {
            sharedM2[threadIdx.y][threadIdx.x] = 0.0;
        }

        __syncthreads();

        for (int i = 0; i < TILE_SIZE; i++) {
            tempSum += sharedM1[threadIdx.y][i] * sharedM2[i][threadIdx.x];
        }

        __syncthreads();
    }

    if (row < rowsA && col < colsB) {
        Result[row * colsB + col] = tempSum;
    }
}

double* createRandomMatrix(int rows, int cols) {
    double* matrix = (double*)malloc(rows * cols * sizeof(double));
    for (int i = 0; i < rows * cols; i++) {
        matrix[i] = (double)rand() / RAND_MAX * 100;
    }
    return matrix;
}

int main() {
    srand(time(NULL));
    int rowsA = 400, 
        colsA = 200, 
        colsB = 300;

    size_t sizeA = rowsA * colsA * sizeof(double);
    size_t sizeB = colsA * colsB * sizeof(double);
    size_t sizeC = rowsA * colsB * sizeof(double);

    double *matrixA = createRandomMatrix(rowsA, colsA);
    double *matrixB = createRandomMatrix(colsA, colsB);
    double *result = (double*)malloc(sizeC);

    double *deviceA, *deviceB, *deviceC;
    cudaMalloc(&deviceA, sizeA);
    cudaMalloc(&deviceB, sizeB);
    cudaMalloc(&deviceC, sizeC);

    cudaMemcpy(deviceA, matrixA, sizeA, cudaMemcpyHostToDevice);
    cudaMemcpy(deviceB, matrixB, sizeB, cudaMemcpyHostToDevice);

    dim3 blockDim(TILE_SIZE, TILE_SIZE);
    dim3 gridDim((colsB + TILE_SIZE - 1) / TILE_SIZE, (rowsA + TILE_SIZE - 1) / TILE_SIZE);

    cudaEvent_t startEvent, stopEvent;
    cudaEventCreate(&startEvent);
    cudaEventCreate(&stopEvent);
    float elapsedMilliseconds = 0;
    float totalSeconds = 0;

    for (int i = 0; i < 10; i++) {
        cudaEventRecord(startEvent);
        matrixMultiply<<<gridDim, blockDim>>>(deviceA, deviceB, deviceC, rowsA, colsA, colsB);
        cudaEventRecord(stopEvent);
        cudaEventSynchronize(stopEvent);

        cudaEventElapsedTime(&elapsedMilliseconds, startEvent, stopEvent);
        totalSeconds += elapsedMilliseconds / 1000.0;
    }

    printf("Average time taken over 10 iterations: %.6f seconds\n", totalSeconds / 10);

    cudaMemcpy(result, deviceC, sizeC, cudaMemcpyDeviceToHost);

    cudaFree(deviceA);
    cudaFree(deviceB);
    cudaFree(deviceC);
    free(matrixA);
    free(matrixB);
    free(result);

    cudaEventDestroy(startEvent);
    cudaEventDestroy(stopEvent);

    return 0;
}

