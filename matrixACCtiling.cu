#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <openacc.h>
#include <sys/time.h>

#define TILE_SIZE 16

double* generateRandomMatrix(int rows, int cols) {
    double* matrix = (double*)malloc(rows * cols * sizeof(double));
    for (int i = 0; i < rows * cols; i++) {
        matrix[i] = (double)rand() / RAND_MAX * 100;  // Random value between 0 and 100
    }
    return matrix;
}

void performMatrixMultiplication(int rowsA, int commonDim, int colsB, double* matA, double* matB, double* result) {
    #pragma acc data copyin(matA[0:rowsA * commonDim], matB[0:commonDim * colsB]) copyout(result[0:rowsA * colsB])
    {
        #pragma acc parallel loop tile(TILE_SIZE , TILE_SIZE )
        for (int i = 0; i < rowsA; i++) {
            for (int j = 0; j < colsB; j++) {
                double tempSum = 0;
                for (int k = 0; k < commonDim; k++) {
                    tempSum += matA[i * commonDim + k] * matB[k * colsB + j];
                }
                result[i * colsB + j] = tempSum;
            }
        }
    }
}


int main() {
    srand(time(NULL));

    int rowsA = 1000, commonDim = 600, colsB = 800; 

    double elapsedCPUTime = 0.0, totalElapsedTime = 0.0;

    for (int iteration = 0; iteration < 10; iteration++) {
        double* matA = generateRandomMatrix(rowsA, commonDim);
        double* matB = generateRandomMatrix(commonDim, colsB);
        double* resultMatrix = (double*)malloc(rowsA * colsB * sizeof(double)); 
      
        struct timeval start, end;
        gettimeofday(&start, NULL);

        performMatrixMultiplication(rowsA, commonDim, colsB, matA, matB, resultMatrix);

        gettimeofday(&end, NULL);

        double elapsedCPUTime = (end.tv_sec - start.tv_sec) + (end.tv_usec - start.tv_usec) / 1000000.0;
        totalElapsedTime += elapsedCPUTime;

        free(matA);
        free(matB);
        free(resultMatrix);
    }

    printf("Average time taken over 10 iterations: %.6f seconds\n", totalElapsedTime / 10);
    return 0;
}
