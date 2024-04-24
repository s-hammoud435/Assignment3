#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <openacc.h>

double* createRandomMatrix(int rows, int cols) {
    double* matrix = (double*)malloc(rows * cols * sizeof(double));
    for (int i = 0; i < rows * cols; i++) {
        matrix[i] = (double)rand() / RAND_MAX * 100;  // Generate random values
    }
    return matrix;
}

void matrixMultiplication(int rowsA, int colsA, int colsB, double* matA, double* matB, double* result) {
    #pragma acc data copyin(matA[0:rowsA * colsA], matB[0:colsA * colsB]) copyout(result[0:rowsA * colsB])
    {
        #pragma acc parallel
        for (int i = 0; i < rowsA; i++) {
            for (int j = 0; j < colsB; j++) {
                double tempSum = 0;
                for (int k = 0; k < colsA; k++) {
                    tempSum += matA[i * colsA + k] * matB[k * colsB + j];
                }
                result[i * colsB + j] = tempSum;
            }
        }
    }
}

int main() {
    srand(time(NULL));

    int rowsA = 1000, colsA = 600, colsB = 800; 

    clock_t startTime, endTime;
    double cpuTimeElapsed = 0.0, totalElapsedTime = 0.0;

    for (int iteration = 0; iteration < 10; iteration++) {
        double* matA = createRandomMatrix(rowsA, colsA);
        double* matB = createRandomMatrix(colsA, colsB);
        double* resultMatrix = (double*)malloc(rowsA * colsB * sizeof(double));  // Output matrix

        startTime = clock();
        matrixMultiplication(rowsA, colsA, colsB, matA, matB, resultMatrix);
        endTime = clock();

        cpuTimeElapsed = ((double)(endTime - startTime)) / CLOCKS_PER_SEC;
        totalElapsedTime += cpuTimeElapsed;

        free(matA);
        free(matB);
        free(resultMatrix);
    }

    printf("Average time taken over 10 iterations: %.6f seconds\n", totalElapsedTime / 10);

    return 0;
}
