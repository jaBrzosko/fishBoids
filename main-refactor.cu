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
#define N 1024 * 4
#define GRID_SIZE 8
#define GRID_RANGE 1    
#define FISH_LENGTH 4.0f
#define FISH_WIDTH 2.0f

#define MAX_VELOCITY 2.0f
#define MIN_VELOCITY 1.2f
#define MAX_ACCELERATION 0.5f

#define SIGHT_ANGLE 3.1415f * 0.55f
#define SIGHT_RANGE 900.0f //squared
#define PROTECTED_RANGE 400.0f // squared

#define TURN_FACTOR 1.5f
#define COHESION_FACTOR 2.0f
#define ALIGNMENT_FACTOR 4.0f
#define SEPARATION_FACTOR 4.0f
////////////////////////////////////////////////////////////////////////////////
//! Parameters
////////////////////////////////////////////////////////////////////////////////

#define mapRange(a1,a2,b1,b2,s) (b1 + (s-a1)*(b2-b1)/(a2-a1))

// constants
const int window_width  = 1000;
const int window_height = 1000;

const int sea_width    = window_width / 2;
const int sea_height   = window_height / 2;

const float cell_width = window_width / GRID_SIZE;
const float cell_height = window_height / GRID_SIZE;

float *d_x, *d_y, *d_vx, *d_vy, *d_future_vx, *d_future_vy;
int *d_gridCell, *d_gridFish, *d_cellStart, *d_startCell, *d_endCell;

// vbo variables
GLuint vbo;
struct cudaGraphicsResource *cuda_vbo_resource;
void *d_vbo_buffer = NULL;

int mouse_old_x, mouse_old_y;
int mouse_buttons = 0;
float rotate_x = 0.0, rotate_y = 0.0;
float translate_z = -1.0;

int fpsCount = 0;        // FPS count for averaging
int fpsLimit = 1;        // FPS limit for sampling
int g_Index = 0;
float avgFPS = 0.0f;
unsigned int frameCount = 0;

StopWatchInterface *timer = NULL;

// declarations
bool initGL(int *argc, char **argv);
void initCUDA();
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
void runCuda(struct cudaGraphicsResource **vbo_resource);


__global__ void setUnsortedGrid(float* x, float* y, int* gridCell, int* gridFish)
{
    unsigned int tid = threadIdx.x + blockIdx.x * blockDim.x;
    int column = (x[tid] + sea_width) / cell_width;
    int row = (y[tid] + sea_height) / cell_height;
    gridCell[tid] = max(min(row * GRID_SIZE + column, GRID_SIZE * GRID_SIZE - 1), 0);
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


__global__ void kernel_normalize_velocity(float *vx, float *vy, float *fvx, float *fvy)
{
    unsigned int tid = threadIdx.x + blockIdx.x * blockDim.x;
    
    float tvx = fvx[tid];
    float tvy = fvy[tid];
    
    float speed = sqrt(tvx * tvx + tvy * tvy);
    if(speed > MAX_VELOCITY)
    {
        tvx = MAX_VELOCITY * tvx / speed;
        tvy = MAX_VELOCITY * tvy / speed;
    }
    else if(speed < MIN_VELOCITY)
    {
        tvx = MIN_VELOCITY * tvx / speed;
        tvy = MIN_VELOCITY * tvy / speed;
    }
    vx[tid] = tvx;
    vy[tid] = tvy;
}

__global__ void prepareStartEndCell(int* cellStart, int* startCell, int* endCell)
{
    unsigned int tid = threadIdx.x + blockIdx.x * blockDim.x;

    if(tid >= GRID_SIZE * GRID_SIZE)
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

    while(startPos < GRID_SIZE * GRID_SIZE && cellStart[startPos] == -1)
        startPos++;
    while(endPos < GRID_SIZE * GRID_SIZE && cellStart[endPos] == -1)
        endPos++;
    



    startCell[tid] = startPos == GRID_SIZE * GRID_SIZE ? N : cellStart[startPos];
    endCell[tid] = endPos == GRID_SIZE * GRID_SIZE ? N : cellStart[endPos];
}

__global__ void kernel_prepare_move(float *x, float *y, float *vx, float *vy, float *fvx, float *fvy,
        int* gridFish, int* startCell, int* endCell, int* gridCell)
{
    unsigned int tid = threadIdx.x + blockIdx.x * blockDim.x;
    int cell = gridCell[tid];
    tid = gridFish[tid];

    // prepare accumulators
    float l_cohesionX = 0, l_cohesionY = 0;
    float l_alignementX = 0, l_alignementY = 0;
    float l_separationX = 0, l_separationY = 0;
    float l_count = 0;

    // prepare variables
    float tvx = vx[tid];
    float tvy = vy[tid];

    float tx = x[tid];
    float ty = y[tid];

    for(int i = -GRID_RANGE; i <= GRID_RANGE; i++)
    {
        int newCell = cell + i * GRID_SIZE;
        if(newCell >= 0 && newCell < GRID_SIZE * GRID_SIZE)
        {
            int startPos = startCell[newCell];
            int endPos = endCell[newCell];

            for(int i = startPos; i < endPos; i++)
            {
                int another = gridFish[i];
                float ax = x[another];
                float ay = y[another];

                float dx = tx - ax;
                float dy = ty - ay;

                float d = dx * dx + dy * dy;
                
                if(d < SIGHT_RANGE && d > 0 && acos((-dx * tvx + -dy * tvy) / sqrt(d * (tvx * tvx + tvy * tvy)) ) < SIGHT_ANGLE)
                {
                    l_cohesionX += ax;
                    l_cohesionY += ay;
                    l_alignementX += vx[another];
                    l_alignementY += vy[another];

                    float dsqrt = sqrt(d);

                    l_separationX += dx / dsqrt;
                    l_separationY += dy / dsqrt;

                    l_count += 1;

                }
            }
        }
    }
    if(l_count > 0)
    {

        float nvx = l_separationX * SEPARATION_FACTOR +
         (l_alignementX / l_count - tvx) * ALIGNMENT_FACTOR + 
         (l_cohesionX / l_count - tx) * COHESION_FACTOR;
        float nvy = l_separationY * SEPARATION_FACTOR + 
         (l_alignementY / l_count - tvy) * ALIGNMENT_FACTOR + 
         (l_cohesionY / l_count - ty) * COHESION_FACTOR;

        float d = sqrt(nvx * nvx + nvy * nvy);

        if(d > 0.001f)
        {
            fvx[tid] = tvx + MAX_ACCELERATION / d * nvx;
            fvy[tid] = tvy + MAX_ACCELERATION / d * nvy;
        }
    }
}

__global__ void kernel_move(float *x, float *y, float *vx, float *vy)
{
    unsigned int tid = threadIdx.x + blockIdx.x * blockDim.x;
    float nx = x[tid] + vx[tid];
    float ny = y[tid] + vy[tid];

    // repair X velocity
    if(nx < -sea_width)
        vx[tid] = vx[tid] + TURN_FACTOR;
    else if(nx > sea_width)
        vx[tid] = vx[tid] - TURN_FACTOR;
    // repair Y velocity
    if(ny < -sea_height)
        vy[tid] = vy[tid] + TURN_FACTOR;
    else if(ny > sea_height)
        vy[tid] = vy[tid] - TURN_FACTOR;
    x[tid] = nx;
    y[tid] = ny;
    
}


__global__ void kernel_display(float *pos, float *x, float *y, float *vx, float *vy)
{
    unsigned int tid = threadIdx.x + blockIdx.x * blockDim.x;
    float p1X = x[tid];
    float p1Y = y[tid];

    float tvx = vx[tid];
    float tvy = vy[tid];

    float coef = FISH_LENGTH / sqrt(tvx * tvx + tvy * tvy);
    float tempX = coef * tvx;
    float tempY = coef * tvy;

    float p2X = p1X - tempX - tempY * FISH_WIDTH / FISH_LENGTH;
    float p2Y = p1Y - tempY + tempX * FISH_WIDTH / FISH_LENGTH;
    float p3X = p1X - tempX + tempY * FISH_WIDTH / FISH_LENGTH;
    float p3Y = p1Y - tempY - tempX * FISH_WIDTH / FISH_LENGTH;

    pos[9 * tid] = p1X / window_width;
    pos[9 * tid + 1] = p1Y / window_height;
    pos[9 * tid + 2] = 0.0f;
    pos[9 * tid + 3] = p2X / window_width;
    pos[9 * tid + 4] = p2Y / window_height;
    pos[9 * tid + 5] = 0.0f;    
    pos[9 * tid + 6] = p3X / window_width;
    pos[9 * tid + 7] = p3Y / window_height;
    pos[9 * tid + 8] = 0.0f;

}


int main(int argc, char **argv)
{
    // Create the CUTIL timer
    sdkCreateTimer(&timer);

    initCUDA();
    initGL(&argc, argv);

    createVBO(&vbo, &cuda_vbo_resource, cudaGraphicsMapFlagsWriteDiscard);
    glutMainLoop();

    return 0;
}

void initCUDA()
{
    float* h_x = new float[N];
    float* h_y = new float[N];
    float* h_vx = new float[N];
    float* h_vy = new float[N];

    // srand(time(NULL));
    srand(0);

    for(int i = 0; i < N; ++i)
    {
        h_x[i] = mapRange(0, 100, -sea_width, sea_width, rand() % 100);
        h_y[i] = mapRange(0, 100, -sea_height, sea_height, rand() % 100);

        // std::cout << h_x[i] << " " << h_y[i] << std::endl;

        h_vx[i] = mapRange(0, 100, -MAX_VELOCITY, MAX_VELOCITY, rand() % 100);
        h_vy[i] = mapRange(0, 100, -MAX_VELOCITY, MAX_VELOCITY, rand() % 100);
        // printf("Line %d has x: %f y: %f vx: %f vy: %f\n", i, h_x[i], h_y[i], h_vx[i], h_vy[i]);
    }
    cudaMalloc(&d_x, N * sizeof(float));
    cudaMalloc(&d_y, N * sizeof(float));
    cudaMalloc(&d_vx, N * sizeof(float));
    cudaMalloc(&d_vy, N * sizeof(float));
    cudaMalloc(&d_future_vx, N * sizeof(float));
    cudaMalloc(&d_future_vy, N * sizeof(float));
    cudaMalloc(&d_gridCell, N * sizeof(int));
    cudaMalloc(&d_gridFish, N * sizeof(int));
    cudaMalloc(&d_cellStart, GRID_SIZE * GRID_SIZE * sizeof(int));
    cudaMalloc(&d_startCell, GRID_SIZE * GRID_SIZE * sizeof(int));
    cudaMalloc(&d_endCell, GRID_SIZE * GRID_SIZE * sizeof(int));

    cudaMemcpy(d_x, h_x, N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_y, h_y, N * sizeof(float), cudaMemcpyHostToDevice);    
    cudaMemcpy(d_vx, h_vx, N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_vy, h_vy, N * sizeof(float), cudaMemcpyHostToDevice);

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

void cleanup()
{

    if (vbo)
    {
        deleteVBO(&vbo, cuda_vbo_resource);
    }
    cudaFree(d_x);
    cudaFree(d_y);
    cudaFree(d_vx);
    cudaFree(d_vy);
    cudaFree(d_future_vx);
    cudaFree(d_future_vy);
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
    runCuda(&cuda_vbo_resource);

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

void runCuda(struct cudaGraphicsResource **vbo_resource)
{
    // map OpenGL buffer object for writing from CUDA
    float *dptr;
    checkCudaErrors(cudaGraphicsMapResources(1, vbo_resource, 0));
    size_t num_bytes;
    checkCudaErrors(cudaGraphicsResourceGetMappedPointer((void **)&dptr, &num_bytes,
                                                         *vbo_resource));
    //printf("CUDA mapped VBO: May access %ld bytes\n", num_bytes);


    // execute the kernel
    dim3 grid2D(N/1024, 1, 1);
    dim3 block2D(1024, 1, 1);
    dim3 gridGridSize(GRID_SIZE * GRID_SIZE / 1024 + 1, 1, 1);
    dim3 blockGridSize(1024, 1, 1);     
    // dim3 grid3D(N / 32, N / 32, 1);
    // dim3 block3D(32, 32, 1);
    // dim3 gridReduce(N * N / 1024, 1, 1);
    // dim3 blockReduce(1024, 1, 1);

    // size_t shm_size = 1024 * sizeof(float);
    // {
    //     float* debug = new float[N];
    //     cudaMemcpy(debug, d_x, N * sizeof(float), cudaMemcpyDeviceToHost);
    //     for(int i = 0; i < N; i++)
    //     {
    //         std::cout << i << ") " << debug[i] << std::endl;
    //     }
    //     exit(1);
    // }
    // Prepare helping grid
    setUnsortedGrid<<<grid2D, block2D>>>(d_x, d_y, d_gridCell, d_gridFish);
    thrust::sort_by_key(thrust::device, d_gridCell, d_gridCell + N, d_gridFish);
    cudaMemset(d_cellStart, -1, GRID_SIZE * GRID_SIZE * sizeof(int));
    prepareCellStart<<<grid2D, block2D>>>(d_gridCell, d_cellStart);
    prepareStartEndCell<<<gridGridSize, blockGridSize>>>(d_cellStart, d_startCell, d_endCell);
    // {
    //     int* debugCellStart = new int[GRID_SIZE * GRID_SIZE];
    //     int* debugStart = new int[GRID_SIZE * GRID_SIZE];
    //     int* debugEnd = new int[GRID_SIZE * GRID_SIZE];
    //     cudaMemcpy(debugCellStart, d_cellStart, GRID_SIZE * GRID_SIZE * sizeof(int), cudaMemcpyDeviceToHost);
    //     cudaMemcpy(debugStart, d_startCell, GRID_SIZE * GRID_SIZE * sizeof(int), cudaMemcpyDeviceToHost);
    //     cudaMemcpy(debugEnd, d_endCell, GRID_SIZE * GRID_SIZE * sizeof(int), cudaMemcpyDeviceToHost);
    //     for(int i = 0; i < GRID_SIZE * GRID_SIZE; i++)
    //     {
    //         std::cout << i << ") fishID: " << debugCellStart[i] << " start: " << debugStart[i] << " end: " << debugEnd[i] << std::endl;
    //     }
    //     exit(1);
    // }
    kernel_prepare_move<<<grid2D, block2D>>>(d_x, d_y, d_vx, d_vy, d_future_vx, d_future_vy, d_gridFish, d_startCell, d_endCell, d_gridCell);
    // {
    //     float* debug = new float[N];
    //     cudaMemcpy(debug, d_count, N * sizeof(float), cudaMemcpyDeviceToHost);
    //     for(int i = 0; i < N; i++)
    //     {
    //         std::cout << i << ") " << debug[i] << std::endl;
    //     }
    //     exit(1);
    // }
    kernel_normalize_velocity<<<grid2D, block2D>>>(d_vx, d_vy, d_future_vx, d_future_vy);
    kernel_move<<<grid2D, block2D>>>(d_x, d_y, d_vx, d_vy);
    kernel_display<<<grid2D, block2D>>>(dptr, d_x, d_y, d_vx, d_vy);

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
        case (27) :
            #if defined(__APPLE__) || defined(MACOSX)
                exit(EXIT_SUCCESS);
            #else
                glutDestroyWindow(glutGetWindow());
                return;
            #endif
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
    sprintf(fps, "Fish simulation: %3.1f fps (Max 100Hz)", avgFPS);
    glutSetWindowTitle(fps);
}