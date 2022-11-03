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

#define MAX_EPSILON_ERROR 10.0f
#define THRESHOLD          0.30f
#define REFRESH_DELAY     10 //ms

////////////////////////////////////////////////////////////////////////////////
//! Parameters
////////////////////////////////////////////////////////////////////////////////
#define N 1024 * 5
#define FISH_LENGTH 8.0f
#define FISH_WIDTH 3.0f

#define MAX_VELOCITY 1.0f
#define MIN_VELOCITY 0.4f

#define MAX_ACCELERATION 0.1f

#define SIGHT_RANGE 225.0f //squared
#define PROTECTED_RANGE 64.0f // squared

#define TURN_FACTOR 1.04f
#define COHESION_FACTOR 0.1f
#define ALIGNMENT_FACTOR 0.1f
#define SEPARATION_FACTOR 0.1f
////////////////////////////////////////////////////////////////////////////////
//! Parameters
////////////////////////////////////////////////////////////////////////////////


#define mapRange(a1,a2,b1,b2,s) (b1 + (s-a1)*(b2-b1)/(a2-a1))

// constants
const int window_width  = 1024;
const int window_height = 960;

const int sea_width    = window_width / 2;
const int sea_height   = window_height / 2;

float *d_x, *d_y, *d_vx, *d_vy, *d_cohesionx, *d_cohesiony, *d_alignmentx, *d_alignmenty, *d_separationx, *d_separationy, *d_count;

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

__global__ void kernel_normalize_velocity(float *vx, float *vy)
{
    unsigned int tid = threadIdx.x + blockIdx.x * blockDim.x;
    
    float tvx = vx[tid];
    float tvy = vy[tid];
    
    float speed = sqrt(tvx * tvx + tvy * tvy);
    if(speed > MAX_VELOCITY)
    {
        vx[tid] = MAX_VELOCITY * tvx / speed;
        vy[tid] = MAX_VELOCITY * tvy / speed;
    }
    else if(speed < MIN_VELOCITY)
    {
        vx[tid] = MIN_VELOCITY * tvx / speed;
        vy[tid] = MIN_VELOCITY * tvy / speed;
    }
}

__global__ void kernel_update_velocity(float *x, float *y, float *vx, float *vy, float *cohX, float *cohY, float *sepX, float *sepY, float *alignX, float *alignY, float *count)
{
    unsigned int tid = threadIdx.x + blockIdx.x * blockDim.x;
    unsigned int ntid = N * tid;

    if(count[ntid] != 0)
    {
        float tvx = vx[tid];
        float tvy = vy[tid];

        int cnt = count[ntid];

        float nvx = sepX[ntid] * SEPARATION_FACTOR + (alignX[ntid] / cnt - tvx) * ALIGNMENT_FACTOR  + (cohX[ntid] / cnt - x[tid]) * COHESION_FACTOR;
        float nvy = sepY[ntid] * SEPARATION_FACTOR + (alignY[ntid] / cnt - tvy) * ALIGNMENT_FACTOR  + (cohY[ntid] / cnt - y[tid]) * COHESION_FACTOR;

        float d = sqrt(nvx * nvx + nvy * nvy);


        vx[tid] = tvx + MAX_ACCELERATION / d * (nvx - tvx);
        vy[tid] = tvy + MAX_ACCELERATION / d * (nvy - tvy);


        // vx[tid] = nvx;
        // vy[tid] = nvy;
    }
}

__global__ void reduce(float *data)
{
    extern __shared__ float sdata[];
    // each thread loads one element from global to shared mem
    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x*blockDim.x + threadIdx.x;
    sdata[tid] = data[i];
    __syncthreads();
    // do reduction in shared mem
    for (unsigned int s=blockDim.x/2; s>0; s>>=1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }
    // write result for this block to global mem
    if (tid == 0) data[blockDim.x * blockIdx.x] = sdata[0];
}

__global__ void finish_reduce(float *data, int count)
{
    unsigned int tid = threadIdx.x + blockDim.x * blockIdx.x;
    int sum = 0;
    for(int i = 0; i < count; i++)
    {
        sum += data[tid * N + i * 1024];
    }
    data[tid * N] = sum;
}

__global__ void kernel_prepare_move(float *x, float *y, float *vx, float *vy, float *cohX, float *cohY, float *sepX, float *sepY, float *alignX, float *alignY, float *count)
{
    unsigned int tidx = threadIdx.x + blockIdx.x * blockDim.x; // Fish number tidx is ANOTHER fish
    unsigned int tidy = threadIdx.y + blockIdx.y * blockDim.y; // Fish number tidy is MY fish

    float dx = x[tidy] - x[tidx];
    float dy = y[tidy] - y[tidx];

    float d = dx * dx + dy * dy;

    unsigned int index = tidy * N + tidx;

    if(d < SIGHT_RANGE && tidx != tidy)
    {
        cohX[index] = x[tidx];
        cohY[index] = y[tidx];
        alignX[index] = vx[tidx];
        alignY[index] = vy[tidx];

        count[index] = 1;
        if(d < PROTECTED_RANGE)
        {
            float dsqrt = sqrt(d);

            sepX[index] = dx / dsqrt;
            sepY[index] = dy / dsqrt;
        }
        else
        {
            sepX[index] = 0;
            sepY[index] = 0;
        }
    }
    else
    {
        cohX[index] = 0;
        cohY[index] = 0;        
        sepX[index] = 0;
        sepY[index] = 0;
        alignX[index] = 0;
        alignY[index] = 0;

        count[index] = 0;
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
        h_x[i] = (rand() % window_width) - sea_width;
        h_y[i] = (rand() % window_height) - sea_height;

        h_vx[i] = mapRange(0, 100, -MAX_VELOCITY, MAX_VELOCITY, rand() % 100);
        h_vy[i] = mapRange(0, 100, -MAX_VELOCITY, MAX_VELOCITY, rand() % 100);
        // printf("Line %d has x: %f y: %f vx: %f vy: %f\n", i, h_x[i], h_y[i], h_vx[i], h_vy[i]);
    }

    cudaMalloc(&d_x, N * sizeof(float));
    cudaMalloc(&d_y, N * sizeof(float));
    cudaMalloc(&d_vx, N * sizeof(float));
    cudaMalloc(&d_vy, N * sizeof(float));
    cudaMalloc(&d_cohesionx, N * N * sizeof(float));
    cudaMalloc(&d_cohesiony, N * N * sizeof(float));    
    cudaMalloc(&d_separationx, N * N * sizeof(float));
    cudaMalloc(&d_separationy, N * N * sizeof(float));
    cudaMalloc(&d_alignmentx, N * N * sizeof(float));
    cudaMalloc(&d_alignmenty, N * N * sizeof(float));
    cudaMalloc(&d_count, N * N * sizeof(float));

    cudaMemcpy(d_x, h_x, N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_y, h_y, N * sizeof(float), cudaMemcpyHostToDevice);    
    cudaMemcpy(d_vx, h_vx, N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_vy, h_vy, N * sizeof(float), cudaMemcpyHostToDevice);
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
    cudaFree(d_separationx);
    cudaFree(d_separationy);
    cudaFree(d_cohesionx);
    cudaFree(d_cohesiony);
    cudaFree(d_alignmentx);
    cudaFree(d_alignmenty);
    cudaFree(d_count);
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
    dim3 grid3D(N / 32, N / 32, 1);
    dim3 block3D(32, 32, 1);
    dim3 gridReduce(N * N / 1024, 1, 1);
    dim3 blockReduce(1024, 1, 1);

    size_t shm_size = 1024 * sizeof(float);

    kernel_prepare_move<<<grid3D, block3D>>>(d_x, d_y, d_vx, d_vy, d_cohesionx, d_cohesiony, d_separationx, d_separationy, d_alignmentx, d_alignmenty, d_count);
    //     {
    //     float *debug = new float[N * N];
    //     cudaMemcpy(debug, d_alignmentx, N * N * sizeof(float), cudaMemcpyDeviceToHost);
    //     for(int i = 0; i < N; i++)
    //     {
    //         std::cout << "start";
    //         for(int j = 0; j < N; j++)
    //         {
    //             std::cout << ";" << debug[i * N + j];
    //         }
    //         std::cout << std::endl;
    //     }
    //     delete[] debug;
    // }
    reduce<<<gridReduce, blockReduce, shm_size>>>(d_cohesionx);
    reduce<<<gridReduce, blockReduce, shm_size>>>(d_cohesiony);
    reduce<<<gridReduce, blockReduce, shm_size>>>(d_separationx);
    reduce<<<gridReduce, blockReduce, shm_size>>>(d_separationy);
    reduce<<<gridReduce, blockReduce, shm_size>>>(d_alignmentx);
    reduce<<<gridReduce, blockReduce, shm_size>>>(d_alignmenty);
    reduce<<<gridReduce, blockReduce, shm_size>>>(d_count);
    if(N / 1024 > 1)
    {
        finish_reduce<<<grid2D, block2D>>>(d_cohesionx, N / 1024);
        finish_reduce<<<grid2D, block2D>>>(d_separationx, N / 1024);
        finish_reduce<<<grid2D, block2D>>>(d_alignmentx, N / 1024);
        finish_reduce<<<grid2D, block2D>>>(d_cohesiony, N / 1024);
        finish_reduce<<<grid2D, block2D>>>(d_separationy, N / 1024);
        finish_reduce<<<grid2D, block2D>>>(d_alignmenty, N / 1024);
        finish_reduce<<<grid2D, block2D>>>(d_count, N / 1024);
    }
    // {
    //     float *debug = new float[N * N];
    //     float *debugX = new float[N * N];
    //     float *debugY = new float[N * N];
    //     cudaMemcpy(debug, d_count, N * N * sizeof(float), cudaMemcpyDeviceToHost);
    //     cudaMemcpy(debugX, d_separationx, N * N * sizeof(float), cudaMemcpyDeviceToHost);
    //     cudaMemcpy(debugY, d_separationx, N * N * sizeof(float), cudaMemcpyDeviceToHost);
    //     for(int i = 0; i < N; i++)
    //     {
    //         // for(int j = 0; j < N; j++)
    //         // {
    //         //     std::cout << "1)" << i << ":" << j << ";" << debug[i * N + j] << std::endl;
    //         // }
    //         std::cout << i << ";" << debug[i * N] << ";" << debugX[i * N] << ";" << debugY[i * N] << std::endl;
    //     }
    //     delete[] debug;
    //     delete[] debugX;
    //     delete[] debugY;
    //     exit(1);
    // }

    // {
    //     float *debugX = new float[N];
    //     float *debugY = new float[N];
    //     cudaMemcpy(debugX, d_vx, N * sizeof(float), cudaMemcpyDeviceToHost);
    //     cudaMemcpy(debugY, d_vy, N * sizeof(float), cudaMemcpyDeviceToHost);
    //     for(int i = 0; i < N; i++)
    //     {
    //         std::cout << "0|" << i << ";" << debugX[i] << ";" << debugY[i] << std::endl;
    //     }
    //     delete[] debugX;
    //     delete[] debugY;
    // }
    kernel_update_velocity<<<grid2D, block2D>>>(d_x, d_y, d_vx, d_vy, d_cohesionx, d_cohesiony, d_separationx, d_separationy, d_alignmentx, d_alignmenty, d_count);
    // {
    //     float *debugX = new float[N];
    //     float *debugY = new float[N];
    //     cudaMemcpy(debugX, d_vx, N * sizeof(float), cudaMemcpyDeviceToHost);
    //     cudaMemcpy(debugY, d_vy, N * sizeof(float), cudaMemcpyDeviceToHost);
    //     for(int i = 0; i < N; i++)
    //     {
    //         std::cout << "1|" << i << ";" << debugX[i] << ";" << debugY[i] << std::endl;
    //     }
    //     delete[] debugX;
    //     delete[] debugY;
    // }
    kernel_normalize_velocity<<<grid2D, block2D>>>(d_vx, d_vy);
    // {
    //     float *debugX = new float[N];
    //     float *debugY = new float[N];
    //     cudaMemcpy(debugX, d_vx, N * sizeof(float), cudaMemcpyDeviceToHost);
    //     cudaMemcpy(debugY, d_vy, N * sizeof(float), cudaMemcpyDeviceToHost);
    //     for(int i = 0; i < N; i++)
    //     {
    //         std::cout << "2|" << i << ";" << debugX[i] << ";" << debugY[i] << std::endl;
    //     }
    //     delete[] debugX;
    //     delete[] debugY;
    //     exit(1);
    // }
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