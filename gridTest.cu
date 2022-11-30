#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>

#include <helper_gl.h>
#include <GL/freeglut.h>

#include <cuda_runtime.h>
#include <cuda_gl_interop.h>

#include <helper_functions.h>
#include <helper_cuda.h>         
#include <vector_types.h>
#include <thrust/sort.h>
#include <thrust/execution_policy.h>

#define N 1024
#define GRID_SIZE 4
#define GRID_RANGE 1

const int window_width  = 100;
const int window_height = 100;
const float cell_width = window_width / GRID_SIZE;
const float cell_height = window_height / GRID_SIZE;

const int sea_width    = window_width / 2;
const int sea_height   = window_height / 2;

__global__ void setUnsortedGrid(float* x, float* y, int* gridCell, int* gridFish)
{
    unsigned int tid = threadIdx.x + blockIdx.x * blockDim.x;
    int column = (x[tid] + sea_width) / cell_width;
    int row = (y[tid] + sea_height) / cell_height;
    gridCell[tid] = row * GRID_SIZE + column;
    gridFish[tid] = tid;

}

__global__ void prepareCellStart(int* gridCell, int* cellStart)
{
    unsigned int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if(tid == 0)
        cellStart[gridCell[0]] = 0;
    else if(gridCell[tid] != gridCell[tid - 1])
        cellStart[gridCell[tid]] = tid;
}

__global__ void prepareStartEndCell(int* cellStart, int* startCell, int* endCell)
{
    unsigned int tid = threadIdx.x + blockIdx.x * blockDim.x;

    if(tid > GRID_SIZE * GRID_SIZE)
        return;

    int startPos = tid - GRID_RANGE;
    int endPos = tid + GRID_RANGE + 1;

    int tidRow = tid / GRID_SIZE;
    if(startPos / GRID_SIZE < tidRow)
        startPos = tidRow * GRID_SIZE;
    if(endPos / GRID_SIZE > tidRow)
        endPos = tidRow * GRID_SIZE + GRID_SIZE;

    if(startPos < 0)
        startPos = 0;
    if(endPos >= GRID_SIZE * GRID_SIZE)
        endPos = GRID_SIZE * GRID_SIZE;

    while(cellStart[startPos] == -1 && startPos < GRID_SIZE * GRID_SIZE)
        startPos++;
    while(cellStart[endPos] == -1 && endPos < GRID_SIZE * GRID_SIZE)
        endPos++;
    



    startCell[tid] = startPos == GRID_SIZE * GRID_SIZE ? N : cellStart[startPos];
    endCell[tid] = endPos == GRID_SIZE * GRID_SIZE ? N : cellStart[endPos];
}


void fixGridHost(float* x, float* y, int* gridCell, int* gridFish)
{
    for(int i = 0; i < N; i++)
    {
        int column = (x[i] + sea_width) / cell_width;
        int row = (y[i] + sea_height) / cell_height;
        gridCell[i] = row * GRID_SIZE + column;
        gridFish[i] = i;
    }

    for(int i = 0; i < N; i++)
    {
        int maxIndex = i;
        for(int j = i + 1; j < N; j++)
        {
            if(gridCell[j] < gridCell[maxIndex])
                maxIndex = j;
        }
        if(maxIndex != i)
        {
            // swamp cell ID
            int temp = gridCell[maxIndex];
            gridCell[maxIndex] = gridCell[i];
            gridCell[i] = temp;
            
            // swap fish ID
            temp = gridFish[maxIndex];
            gridFish[maxIndex] = gridFish[i];
            gridFish[i] = temp;
        }
    }
}

void printGrid(int* gridCell, int* gridFish)
{
    for(int i = 0; i < N; i++)
    {
            std::cout << i << " (" << gridCell[i] << ", " << gridFish[i] << ")" << std::endl;
    }
}

int main(int argc, char** argv)
{
    float* h_x = new float[N];
    float* h_y = new float[N];
    int* h_gridCell = new int[N];
    int* h_gridFish = new int[N];
    int* h_cellStart = new int[GRID_SIZE * GRID_SIZE];
    int* h_startCell = new int[GRID_SIZE * GRID_SIZE];
    int* h_endCell = new int[GRID_SIZE * GRID_SIZE];


    // srand(time(NULL));
    srand(0);

    for(int i = 0; i < N; ++i)
    {
        h_x[i] = (rand() % window_width) - sea_width;
        h_y[i] = (rand() % window_height) - sea_height;
        // printf("Line %d has x: %f y: %f vx: %f vy: %f\n", i, h_x[i], h_y[i], h_vx[i], h_vy[i]);
    }
    float *d_x, *d_y;
    int *d_gridCell, *d_gridFish, *d_cellStart, *d_startCell, *d_endCell;
    
    cudaMalloc(&d_x, N * sizeof(float));
    cudaMalloc(&d_y, N * sizeof(float));
    cudaMalloc(&d_gridCell, N * sizeof(int));
    cudaMalloc(&d_gridFish, N * sizeof(int));
    cudaMalloc(&d_cellStart, GRID_SIZE * GRID_SIZE * sizeof(int));
    cudaMalloc(&d_startCell, GRID_SIZE * GRID_SIZE * sizeof(int));
    cudaMalloc(&d_endCell, GRID_SIZE * GRID_SIZE * sizeof(int));
    cudaMemcpy(d_x, h_x, N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_y, h_y, N * sizeof(float), cudaMemcpyHostToDevice);


    fixGridHost(h_x, h_y, h_gridCell, h_gridFish);

    std::cout << "CPU:" << std::endl;
    printGrid(h_gridCell, h_gridFish);

    int threads = min(1024, N);
    int blocks = max(1, N / 1024);

    setUnsortedGrid<<<blocks, threads>>>(d_x, d_y, d_gridCell, d_gridFish);
    thrust::sort_by_key(thrust::device, d_gridCell, d_gridCell + N, d_gridFish);
    
    cudaMemset(d_cellStart, -1, GRID_SIZE * GRID_SIZE * sizeof(int));
    prepareCellStart<<<blocks, threads>>>(d_gridCell, d_cellStart);

    // threads = 1024;
    // blocks = GRID_SIZE * GRID_SIZE / 1024 + 1;
    threads = GRID_SIZE * GRID_SIZE;
    blocks = 1;
    prepareStartEndCell<<<blocks, threads>>>(d_cellStart, d_startCell, d_endCell);

    cudaMemcpy(h_gridCell, d_gridCell, N * sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_gridFish, d_gridFish, N * sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_cellStart, d_cellStart, GRID_SIZE * GRID_SIZE * sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_startCell, d_startCell, GRID_SIZE * GRID_SIZE * sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_endCell, d_endCell, GRID_SIZE * GRID_SIZE * sizeof(int), cudaMemcpyDeviceToHost);


    std::cout << "CUDA:" << std::endl;
    printGrid(h_gridCell, h_gridFish);

    std::cout << "Cell start:" << std::endl;
    for(int i = 0; i < GRID_SIZE * GRID_SIZE; i++)
    {
        std::cout << i << ": " << h_cellStart[i] << std::endl;
    }
        
    std::cout << "Cell range:" << std::endl;
    for(int i = 0; i < GRID_SIZE * GRID_SIZE; i++)
    {
        std::cout << i << ") start:" << h_startCell[i] << " end: " << h_endCell[i] << std::endl;
    }

    return 0;
}