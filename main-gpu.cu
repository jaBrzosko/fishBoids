// ========================= IMPORTANT =========================
// The graphics part of this code strongly rely on NVIDIA's code
// Therefore the below notice will be posted in my code
// =============================================================

/* Copyright (c) 2021, NVIDIA CORPORATION. All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 *  * Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 *  * Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the distribution.
 *  * Neither the name of NVIDIA CORPORATION nor the names of its
 *    contributors may be used to endorse or promote products derived
 *    from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
 * EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
 * PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
 * CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
 * EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
 * PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
 * PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
 * OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

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

#define MAX_EPSILON_ERROR 10.0f
#define THRESHOLD          0.30f
#define REFRESH_DELAY     10 //ms

////////////////////////////////////////////////////////////////////////////////
//! Parameters
////////////////////////////////////////////////////////////////////////////////
unsigned int N = 1024 * 20;
#define GRID_SIZE 20
#define GRID_RANGE 1
#define FISH_LENGTH 8.0f
#define FISH_WIDTH 4.0f

#define MAX_VELOCITY 5.0f
#define MIN_VELOCITY 4.2f
#define MAX_ACCELERATION 1.0f

#define SIGHT_ANGLE 3.1415f * 0.55f
#define SIGHT_RANGE 900.0f //squared

#define TURN_FACTOR 20.5f
#define INCREMENT_FACTOR 0.5f
////////////////////////////////////////////////////////////////////////////////
//! Parameters
////////////////////////////////////////////////////////////////////////////////

#define mapRange(a1,a2,b1,b2,s) (b1 + (s-a1)*(b2-b1)/(a2-a1))

// constants
const int window_width  = 1000;
const int window_height = 1000;
const int window_depth = 1000;

const int sea_width    = window_width / 2;
const int sea_height   = window_height / 2;
const int sea_depth   = window_depth / 2;

const float cell_width = window_width / GRID_SIZE;
const float cell_height = window_height / GRID_SIZE;

// tabs used for grid optimization
int *d_gridCell, *d_gridFish, *d_cellStart, *d_startCell, *d_endCell;

bool doAnimate = true; // operates pause button

struct Constants
{
    float cohesion_factor;
    float alignment_factor;
    float separation_factor;
    Constants(float coh, float align, float sep)
    {
        cohesion_factor = coh;
        alignment_factor = align;
        separation_factor = sep;
    }
};
Constants *d_constants, *h_constants;

struct FishData
{
    // position
    float *x;
    float *y;
    float *z;

    // velocity
    float *vx;
    float *vy;
    float *vz;

    // future velocity - used to update model synchronously
    float *fvx;
    float *fvy;
    float *fvz;
};

FishData *d_fishData;


// vbo variables
GLuint vbo;
struct cudaGraphicsResource *cuda_vbo_resource;
void *d_vbo_buffer = NULL;

int mouse_old_x, mouse_old_y;
int mouse_buttons = 0;
float rotate_x = 35, rotate_y = 35;
float translate_z = -2.0;

int fpsCount = 0;        // FPS count for averaging
int fpsLimit = 1;        // FPS limit for sampling
int g_Index = 0;
float avgFPS = 0.0f;
unsigned int frameCount = 0;

StopWatchInterface *timer = NULL;

// declarations
bool initGL(int *argc, char **argv);
void initMemory();
void createVBO(GLuint *vbo, struct cudaGraphicsResource **vbo_res,
               unsigned int vbo_res_flags);
void deleteVBO(GLuint *vbo, struct cudaGraphicsResource *vbo_res);

void cleanup();
void keyboard(unsigned char key, int x, int y);
void mouse(int button, int state, int x, int y);
void motion(int x, int y);

void computeFPS();
void timerEvent(int value);
void display();
void runAnimation(struct cudaGraphicsResource **vbo_resource);

// calculates each fish position in the grid
__global__ void setUnsortedGrid(FishData *data, int* gridCell, int* gridFish)
{
    unsigned int tid = threadIdx.x + blockIdx.x * blockDim.x;
    int column = (data->x[tid] + sea_width) / cell_width;
    int row = (data->y[tid] + sea_height) / cell_height;
    gridCell[tid] = max(min(row * GRID_SIZE + column, GRID_SIZE * GRID_SIZE - 1), 0);
    gridFish[tid] = tid;

}

// if gridCell changes value over neighbors update cellStart
__global__ void prepareCellStart(int* gridCell, int* cellStart)
{
    unsigned int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if(tid == 0)
        cellStart[gridCell[0]] = 0;
    else if(gridCell[tid] != gridCell[tid - 1])
        cellStart[gridCell[tid]] = tid;
}

// fish should not be going neither too fast nor too slow
__global__ void kernel_normalize_velocity(FishData *data)
{
    unsigned int tid = threadIdx.x + blockIdx.x * blockDim.x;
    
    float tvx = data->fvx[tid];
    float tvy = data->fvy[tid];
    float tvz = data->fvz[tid];
    
    float speed = sqrt(tvx * tvx + tvy * tvy + tvz * tvz);
    if(speed > MAX_VELOCITY)
    {
        tvx = MAX_VELOCITY * tvx / speed;
        tvy = MAX_VELOCITY * tvy / speed;
        tvz = MAX_VELOCITY * tvz / speed;
    }
    else if(speed < MIN_VELOCITY)
    {
        tvx = MIN_VELOCITY * tvx / speed;
        tvy = MIN_VELOCITY * tvy / speed;
        tvz = MIN_VELOCITY * tvz / speed;
    }
    data->vx[tid] = tvx;
    data->vy[tid] = tvy;
    data->vz[tid] = tvz;
}

// In order not to calculate first and last fish index for each cell we precompute it
__global__ void prepareStartEndCell(int* cellStart, int* startCell, int* endCell, unsigned int N)
{
    unsigned int tid = threadIdx.x + blockIdx.x * blockDim.x;

    if(tid >= GRID_SIZE * GRID_SIZE)
        return;

    int startPos = tid - GRID_RANGE;
    int endPos = tid + GRID_RANGE + 1;

    // check whether start and end is in the same row
    int tidRow = tid / GRID_SIZE;
    if(startPos / GRID_SIZE < tidRow)
        startPos = tidRow * GRID_SIZE;
    if(endPos / GRID_SIZE > tidRow)
        endPos = tidRow * GRID_SIZE + GRID_SIZE;

    // start and end have to be in the grid
    if(startPos < 0)
        startPos = 0;
    if(endPos >= GRID_SIZE * GRID_SIZE)
        endPos = GRID_SIZE * GRID_SIZE;

    // ommiting empty cells -> those have -1
    while(startPos < GRID_SIZE * GRID_SIZE && cellStart[startPos] == -1)
        startPos++;
    while(endPos < GRID_SIZE * GRID_SIZE && cellStart[endPos] == -1)
        endPos++;

    startCell[tid] = startPos == GRID_SIZE * GRID_SIZE ? N : cellStart[startPos];
    endCell[tid] = endPos == GRID_SIZE * GRID_SIZE ? N : cellStart[endPos];
}


// Each fish checks its neighbors depending on the grid
__global__ void kernel_prepare_move(FishData *data,
        int* gridFish, int* startCell, int* endCell, int* gridCell,
        struct Constants *consts)
{
    unsigned int tid = threadIdx.x + blockIdx.x * blockDim.x;
    int cell = gridCell[tid];
    tid = gridFish[tid];

    // prepare accumulators
    float l_cohesionX = 0, l_cohesionY = 0, l_cohesionZ = 0;
    float l_alignementX = 0, l_alignementY = 0, l_alignementZ = 0;
    float l_separationX = 0, l_separationY = 0, l_separationZ = 0;
    float l_count = 0;

    // prepare variables to reduce reads
    float tvx = data->vx[tid];
    float tvy = data->vy[tid];
    float tvz = data->vz[tid];

    float tx = data->x[tid];
    float ty = data->y[tid];
    float tz = data->z[tid];

    for(int i = -GRID_RANGE; i <= GRID_RANGE; i++)
    {
        int newCell = cell + i * GRID_SIZE;
        if(newCell >= 0 && newCell < GRID_SIZE * GRID_SIZE)
        {
            // index of first and last fish which we will have to check
            int startPos = startCell[newCell];
            int endPos = endCell[newCell];

            for(int i = startPos; i < endPos; i++)
            {
                int another = gridFish[i];
                float ax = data->x[another];
                float ay = data->y[another];
                float az = data->z[another];

                float dx = tx - ax;
                float dy = ty - ay;
                float dz = tz - az;

                float d = dx * dx + dy * dy + dz * dz;
                
                // if another fish is close enough and in view angle we apply boid computations
                if(d < SIGHT_RANGE && d > 0 && acos((-dx * tvx + -dy * tvy + -dz * tvz) / sqrt(d * (tvx * tvx + tvy * tvy + tvz * tvz)) ) < SIGHT_ANGLE)
                {
                    l_cohesionX += ax;
                    l_cohesionY += ay;
                    l_cohesionZ += az;

                    l_alignementX += data->vx[another];
                    l_alignementY += data->vy[another];
                    l_alignementZ += data->vz[another];

                    float dsqrt = sqrt(d);

                    l_separationX += dx / dsqrt;
                    l_separationY += dy / dsqrt;
                    l_separationZ += dz / dsqrt;

                    l_count += 1;

                }
            }
        }
    }

    // after all computations if fish neighbors any other fish its velocity will get updated (in the future via fv)
    if(l_count > 0)
    {

        float nvx = l_separationX * consts->separation_factor +
         (l_alignementX / l_count - tvx) * consts->alignment_factor + 
         (l_cohesionX / l_count - tx) * consts->separation_factor;
        float nvy = l_separationY * consts->separation_factor + 
         (l_alignementY / l_count - tvy) * consts->alignment_factor + 
         (l_cohesionY / l_count - ty) * consts->separation_factor;
        float nvz = l_separationZ * consts->separation_factor + 
         (l_alignementZ / l_count - tvz) * consts->alignment_factor + 
         (l_cohesionZ / l_count - tz) * consts->separation_factor;

        float d = sqrt(nvx * nvx + nvy * nvy + nvz * nvz);

        if(d > 0.001f)
        {
            data->fvx[tid] = tvx + MAX_ACCELERATION / d * nvx;
            data->fvy[tid] = tvy + MAX_ACCELERATION / d * nvy;
            data->fvz[tid] = tvz + MAX_ACCELERATION / d * nvz;
        }
    }
}

// Apply velocity to position
__global__ void kernel_move(FishData *data)
{
    unsigned int tid = threadIdx.x + blockIdx.x * blockDim.x;
    float nx = data->x[tid] + data->vx[tid];
    float ny = data->y[tid] + data->vy[tid];
    float nz = data->z[tid] + data->vz[tid];

    // repair X velocity
    if(nx < -sea_width)
        data->vx[tid] = -data->vx[tid];// + TURN_FACTOR;
    else if(nx > sea_width)
        data->vx[tid] = -data->vx[tid];// - TURN_FACTOR;
    // repair Y velocity
    if(ny < -sea_height)
        data->vy[tid] = -data->vy[tid];// + TURN_FACTOR;
    else if(ny > sea_height)
        data->vy[tid] = -data->vy[tid];// - TURN_FACTOR;
    // repair Z velocity
    if(nz < -sea_depth)
        data->vz[tid] = -data->vz[tid];// + TURN_FACTOR;
    else if(nz > sea_depth)
        data->vz[tid] = -data->vz[tid];// - TURN_FACTOR;

    // update position
    data->x[tid] = data->x[tid] + data->vx[tid];
    data->y[tid] = data->y[tid] + data->vy[tid];
    data->z[tid] = data->z[tid] + data->vz[tid];
}

// display function - draws triangles
__global__ void kernel_display(float *pos, FishData *data)
{
    unsigned int tid = threadIdx.x + blockIdx.x * blockDim.x;

    float tvx = data->vx[tid];
    float tvy = data->vy[tid];
    float tvz = data->vz[tid];

    float coef = FISH_LENGTH / sqrt(tvx * tvx + tvy * tvy + tvz * tvz);
    float tempX = coef * tvx;
    float tempY = coef * tvy;
    float tempZ = coef * tvz;

    float p1X = data->x[tid];
    float p1Y = data->y[tid];
    float p1Z = data->z[tid];

    float p2X = p1X;
    float p2Y = p1Y + tempZ * FISH_WIDTH / FISH_LENGTH;
    float p2Z = p1Z - tempY * FISH_WIDTH / FISH_LENGTH;

    float p3X = p1X;
    float p3Y = p1Y - tempZ * FISH_WIDTH / FISH_LENGTH;
    float p3Z = p1Z + tempY * FISH_WIDTH / FISH_LENGTH;

    p1X += tempX;
    p1Y += tempY;
    p1Z += tempZ;

    pos[9 * tid] = p1X / window_width;
    pos[9 * tid + 1] = p1Y / window_height;
    pos[9 * tid + 2] = p1Z / window_depth;
    pos[9 * tid + 3] = p2X / window_width;
    pos[9 * tid + 4] = p2Y / window_height;
    pos[9 * tid + 5] = p2Z / window_depth;
    pos[9 * tid + 6] = p3X / window_width;
    pos[9 * tid + 7] = p3Y / window_height;
    pos[9 * tid + 8] = p3Z / window_depth;

}


int main(int argc, char **argv)
{
    // Parameters input
    h_constants = new Constants(4.0f, 4.0f, 4.0f);
    if(argc > 1)
    {
        N = 1024 * atoi(argv[1]);
        if(!N)
        {
            std::cerr << "Parameter for N is not valid!" << std::endl;
            exit(1);
        }
    }
    if(argc == 5)
    {
        float coh = atof(argv[2]);
        float ali = atof(argv[3]);
        float sep = atof(argv[4]);
        if(coh <= 0 || ali <= 0 || sep <= 0)
        {
            std::cerr << "One of the parameters is not valid! They have to be floats and be positive" << std::endl;
            exit(1);
        }
        std::cout << "Coh: " << coh << " Ali: " << ali << " Sep: " << sep << std::endl;
    }
    // Create the CUTIL timer
    sdkCreateTimer(&timer);

    initMemory();
    int targc = 1;
    char** targv = new char*[1];
    targv[0] = argv[0];
    initGL(&targc, targv);

    createVBO(&vbo, &cuda_vbo_resource, cudaGraphicsMapFlagsWriteDiscard);
    glutMainLoop();

    return 0;
}

// All memory needed for GPU (and some for CPU)
void initMemory()
{
    float* h_x = new float[N];
    float* h_y = new float[N];
    float* h_z = new float[N];
    float* h_vx = new float[N];
    float* h_vy = new float[N];
    float* h_vz = new float[N];

    cudaMalloc(&d_constants, sizeof(Constants));
    cudaMemcpy(d_constants, h_constants, sizeof(Constants), cudaMemcpyHostToDevice);
    srand(time(NULL));

    for(int i = 0; i < N; ++i)
    {
        h_x[i] = mapRange(0, 100, -sea_width, sea_width, rand() % 100);
        h_y[i] = mapRange(0, 100, -sea_height, sea_height, rand() % 100);
        h_z[i] = mapRange(0, 100, -sea_depth, sea_depth, rand() % 100);

        h_vx[i] = mapRange(0, 100, -MAX_VELOCITY, MAX_VELOCITY, rand() % 100);
        h_vy[i] = mapRange(0, 100, -MAX_VELOCITY, MAX_VELOCITY, rand() % 100);
        h_vz[i] = mapRange(0, 100, -MAX_VELOCITY, MAX_VELOCITY, rand() % 100);
    }

    // Since d_fishData is pointer to struct it has some more steps to be allocated
    float *t_x, *t_y, *t_z;
    float *t_vx, *t_vy, *t_vz;
    float *t_fvx, *t_fvy, *t_fvz;
    cudaMalloc(&t_x, N * sizeof(float));
    cudaMalloc(&t_y, N * sizeof(float));
    cudaMalloc(&t_z, N * sizeof(float));
    cudaMalloc(&t_vx, N * sizeof(float));
    cudaMalloc(&t_vy, N * sizeof(float));
    cudaMalloc(&t_vz, N * sizeof(float));
    cudaMalloc(&t_fvx, N * sizeof(float));
    cudaMalloc(&t_fvy, N * sizeof(float));
    cudaMalloc(&t_fvz, N * sizeof(float));

    cudaMemcpy(t_x, h_x, N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(t_y, h_y, N * sizeof(float), cudaMemcpyHostToDevice);    
    cudaMemcpy(t_z, h_z, N * sizeof(float), cudaMemcpyHostToDevice);    
    cudaMemcpy(t_vx, h_vx, N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(t_vy, h_vy, N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(t_vz, h_vz, N * sizeof(float), cudaMemcpyHostToDevice);

    FishData *temp_data = new FishData();
    temp_data->x = t_x;
    temp_data->y = t_y;
    temp_data->z = t_z;

    temp_data->vx = t_vx;
    temp_data->vy = t_vy;
    temp_data->vz = t_vz;

    temp_data->fvx = t_fvx;
    temp_data->fvy = t_fvy;
    temp_data->fvz = t_fvz;

    cudaMalloc(&d_fishData, sizeof(FishData));
    cudaMemcpy(d_fishData, temp_data, sizeof(FishData), cudaMemcpyHostToDevice);
    cudaMalloc(&d_gridCell, N * sizeof(int));
    cudaMalloc(&d_gridFish, N * sizeof(int));
    cudaMalloc(&d_cellStart, GRID_SIZE * GRID_SIZE * sizeof(int));
    cudaMalloc(&d_startCell, GRID_SIZE * GRID_SIZE * sizeof(int));
    cudaMalloc(&d_endCell, GRID_SIZE * GRID_SIZE * sizeof(int));

    free(h_x);
    free(h_y);
    free(h_vx);
    free(h_vy);
}

bool initGL(int *argc, char **argv)
{
    glutInit(argc, argv);
    glutInitDisplayMode(GLUT_RGBA | GLUT_DOUBLE);
    glutInitWindowSize(window_width, window_height);
    glutCreateWindow("Fish simulation");
    glutDisplayFunc(display);
    glutKeyboardFunc(keyboard);
    glutMotionFunc(motion);
    glutTimerFunc(REFRESH_DELAY, timerEvent,0);

    // initialize necessary OpenGL extensions
    if (! isGLVersionSupported(2,0))
    {
        fprintf(stderr, "ERROR: Support for necessary OpenGL extensions missing.");
        fflush(stderr);
        return false;
    }

    // default initialization
    glClearColor(0.0, 0.0, 0.0, 1.0);
    glDisable(GL_DEPTH_TEST);

    // viewport
    glViewport(0, 0, window_width, window_height);

    // projection
    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    gluPerspective(60.0, (GLfloat)window_width / (GLfloat) window_height, 0.1, 10.0);

    SDK_CHECK_ERROR_GL();

    glutMouseFunc(mouse);
    glutCloseFunc(cleanup);


    return true;
}

void timerEvent(int value)
{
    if (glutGetWindow())
    {
        glutPostRedisplay();
        glutTimerFunc(REFRESH_DELAY, timerEvent,0);
    }
}

// Free memory
void cleanup()
{

    if (vbo)
    {
        deleteVBO(&vbo, cuda_vbo_resource);
    }
    FishData *data = new FishData();
    cudaMemcpy(data, d_fishData, sizeof(FishData), cudaMemcpyDeviceToHost);
    cudaFree(data->x);
    cudaFree(data->y);
    cudaFree(data->z);
    cudaFree(data->vx);
    cudaFree(data->vy);
    cudaFree(data->vz);
    cudaFree(data->vx);
    cudaFree(data->vy);
    cudaFree(data->vz);
    cudaFree(d_fishData);
    cudaFree(d_constants);
    delete h_constants;
    delete data;
}

void createVBO(GLuint *vbo, struct cudaGraphicsResource **vbo_res,
               unsigned int vbo_res_flags)
{
    assert(vbo);

    // create buffer object
    glGenBuffers(1, vbo);
    glBindBuffer(GL_ARRAY_BUFFER, *vbo);

    // initialize buffer object
    unsigned int size = 9 * N * sizeof(float);
    glBufferData(GL_ARRAY_BUFFER, size, 0, GL_DYNAMIC_DRAW);

    glBindBuffer(GL_ARRAY_BUFFER, 0);

    // register this buffer object with CUDA
    checkCudaErrors(cudaGraphicsGLRegisterBuffer(vbo_res, *vbo, vbo_res_flags));

    SDK_CHECK_ERROR_GL();
}

void deleteVBO(GLuint *vbo, struct cudaGraphicsResource *vbo_res)
{

    // unregister this buffer object with CUDA
    checkCudaErrors(cudaGraphicsUnregisterResource(vbo_res));

    glBindBuffer(1, *vbo);
    glDeleteBuffers(1, vbo);

    *vbo = 0;
}

void display()
{
    sdkStartTimer(&timer);

    // run CUDA kernel to generate vertex positions
    runAnimation(&cuda_vbo_resource);

    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    // set view matrix
    glMatrixMode(GL_MODELVIEW);
    glLoadIdentity();
    glTranslatef(0.0, 0.0, translate_z);
    glRotatef(rotate_x, 1.0, 0.0, 0.0);
    glRotatef(rotate_y, 0.0, 1.0, 0.0);

    // render from the vbo
    glBindBuffer(GL_ARRAY_BUFFER, vbo);
    glVertexPointer(3, GL_FLOAT, 0, 0);

    glEnableClientState(GL_VERTEX_ARRAY);
    glColor3f(1.0, 0.0, 0.0);
    glDrawArrays(GL_TRIANGLES, 0, 9 * N);
    glDisableClientState(GL_VERTEX_ARRAY);

    glutSwapBuffers();
    sdkStopTimer(&timer);
    computeFPS();
}

// Main part of this program.
// Updates grid and then applies boid algorithm
void runAnimation(struct cudaGraphicsResource **vbo_resource)
{
    // map OpenGL buffer object for writing from CUDA
    float *dptr;
    checkCudaErrors(cudaGraphicsMapResources(1, vbo_resource, 0));
    size_t num_bytes;
    checkCudaErrors(cudaGraphicsResourceGetMappedPointer((void **)&dptr, &num_bytes,
                                                         *vbo_resource));
    dim3 grid2D(N/1024, 1, 1);
    dim3 block2D(1024, 1, 1);
    dim3 gridGridSize(GRID_SIZE * GRID_SIZE / 1024 + 1, 1, 1);
    dim3 blockGridSize(1024, 1, 1);
    if(doAnimate)
    {     
        // Prepare helping grid
        setUnsortedGrid<<<grid2D, block2D>>>(d_fishData, d_gridCell, d_gridFish);
        thrust::sort_by_key(thrust::device, d_gridCell, d_gridCell + N, d_gridFish);
        cudaMemset(d_cellStart, -1, GRID_SIZE * GRID_SIZE * sizeof(int));
        prepareCellStart<<<grid2D, block2D>>>(d_gridCell, d_cellStart);
        prepareStartEndCell<<<gridGridSize, blockGridSize>>>(d_cellStart, d_startCell, d_endCell, N);

        // Start proper move/animation
        kernel_prepare_move<<<grid2D, block2D>>>(d_fishData, d_gridFish, d_startCell, d_endCell, d_gridCell, d_constants);
        kernel_normalize_velocity<<<grid2D, block2D>>>(d_fishData);
        kernel_move<<<grid2D, block2D>>>(d_fishData);
    }
    // Prepar data to be dsiplayed
    kernel_display<<<grid2D, block2D>>>(dptr, d_fishData);

    // unmap buffer object
    checkCudaErrors(cudaGraphicsUnmapResources(1, vbo_resource, 0));
}

////////////////////////////////////////////////////////////////////////////////
//! Keyboard events handler
////////////////////////////////////////////////////////////////////////////////
void keyboard(unsigned char key, int /*x*/, int /*y*/)
{
    switch (key)
    {
        case (27) : // escape - exit
            #if defined(__APPLE__) || defined(MACOSX)
                exit(EXIT_SUCCESS);
            #else
                glutDestroyWindow(glutGetWindow());
                return;
            #endif
            break;
        case (' '): //space - pause
            doAnimate = !doAnimate;
            break;
        case ('q'):
            // add cohesion
            h_constants->cohesion_factor += INCREMENT_FACTOR;
            cudaMemcpy(d_constants, h_constants, sizeof(Constants), cudaMemcpyHostToDevice);
            std::cout << "New cohesion factor: " << h_constants->cohesion_factor << std::endl;
            break;
        case ('a'):
            // sub cohesion
            h_constants->cohesion_factor -= INCREMENT_FACTOR;
            cudaMemcpy(d_constants, h_constants, sizeof(Constants), cudaMemcpyHostToDevice);
            std::cout << "New cohesion factor: " << h_constants->cohesion_factor << std::endl;
            break;
        case ('w'):
            // add alignment
            h_constants->alignment_factor += INCREMENT_FACTOR;
            cudaMemcpy(d_constants, h_constants, sizeof(Constants), cudaMemcpyHostToDevice);
            std::cout << "New alignment factor: " << h_constants->alignment_factor << std::endl;
            break;
        case ('s'):
            // sub alignment
            h_constants->alignment_factor -= INCREMENT_FACTOR;
            cudaMemcpy(d_constants, h_constants, sizeof(Constants), cudaMemcpyHostToDevice);
            std::cout << "New alignment factor: " << h_constants->alignment_factor << std::endl;
            break;
        case ('e'):
            // add separation
            h_constants->separation_factor += INCREMENT_FACTOR;
            cudaMemcpy(d_constants, h_constants, sizeof(Constants), cudaMemcpyHostToDevice);
            std::cout << "New separation factor: " << h_constants->separation_factor << std::endl;
            break;
        case ('d'):
            // sub separation
            h_constants->separation_factor -= INCREMENT_FACTOR;
            cudaMemcpy(d_constants, h_constants, sizeof(Constants), cudaMemcpyHostToDevice);
            std::cout << "New separation factor: " << h_constants->separation_factor << std::endl;
            break;
    }
}

////////////////////////////////////////////////////////////////////////////////
//! Mouse event handlers
////////////////////////////////////////////////////////////////////////////////
void mouse(int button, int state, int x, int y)
{
    if (state == GLUT_DOWN)
    {
        mouse_buttons |= 1<<button;
    }
    else if (state == GLUT_UP)
    {
        mouse_buttons = 0;
    }

    mouse_old_x = x;
    mouse_old_y = y;
}

void motion(int x, int y)
{
    float dx, dy;
    dx = (float)(x - mouse_old_x);
    dy = (float)(y - mouse_old_y);

    if (mouse_buttons & 1)
    {
        rotate_x += dy * 0.2f;
        rotate_y += dx * 0.2f;
    }
    else if (mouse_buttons & 4)
    {
        translate_z += dy * 0.01f;
    }

    mouse_old_x = x;
    mouse_old_y = y;
}

void computeFPS()
{
    frameCount++;
    fpsCount++;

    if (fpsCount == fpsLimit)
    {
        avgFPS = 1.f / (sdkGetAverageTimerValue(&timer) / 1000.f);
        fpsCount = 0;
        fpsLimit = (int)MAX(avgFPS, 1.f);

        sdkResetTimer(&timer);
    }

    char fps[256];
    sprintf(fps, "Fish simulation: %3.1f fps", avgFPS);
    glutSetWindowTitle(fps);
}