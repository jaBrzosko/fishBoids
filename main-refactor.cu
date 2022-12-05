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

#define TURN_FACTOR 10.5f
////////////////////////////////////////////////////////////////////////////////
//! Parameters
////////////////////////////////////////////////////////////////////////////////

#define mapRange(a1,a2,b1,b2,s) (b1 + (s-a1)*(b2-b1)/(a2-a1))

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

// constants
const int window_width  = 1000;
const int window_height = 1000;
const int window_depth = 1000;

const int sea_width    = window_width / 2;
const int sea_height   = window_height / 2;
const int sea_depth   = window_depth / 2;

const float cell_width = window_width / GRID_SIZE;
const float cell_height = window_height / GRID_SIZE;

float *d_x, *d_y, *d_z, *d_vx, *d_vy, *d_vz, *d_future_vx, *d_future_vy, *d_future_vz;
int *d_gridCell, *d_gridFish, *d_cellStart, *d_startCell, *d_endCell;

bool doAnimate = true;

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


__global__ void kernel_normalize_velocity(float *vx, float *vy, float *vz, float *fvx, float *fvy, float *fvz)
{
    unsigned int tid = threadIdx.x + blockIdx.x * blockDim.x;
    
    float tvx = fvx[tid];
    float tvy = fvy[tid];
    float tvz = fvz[tid];
    
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
    vx[tid] = tvx;
    vy[tid] = tvy;
    vz[tid] = tvz;
}

__global__ void prepareStartEndCell(int* cellStart, int* startCell, int* endCell, unsigned int N)
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

__global__ void kernel_prepare_move(float *x, float *y, float *z, 
        float *vx, float *vy, float *vz,
        float *fvx, float *fvy, float * fvz,
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

    // prepare variables
    float tvx = vx[tid];
    float tvy = vy[tid];
    float tvz = vz[tid];

    float tx = x[tid];
    float ty = y[tid];
    float tz = z[tid];

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
                float az = z[another];

                float dx = tx - ax;
                float dy = ty - ay;
                float dz = tz - az;

                float d = dx * dx + dy * dy + dz * dz;
                
                if(d < SIGHT_RANGE && d > 0 && acos((-dx * tvx + -dy * tvy + -dz * tvz) / sqrt(d * (tvx * tvx + tvy * tvy + tvz * tvz)) ) < SIGHT_ANGLE)
                {
                    l_cohesionX += ax;
                    l_cohesionY += ay;
                    l_cohesionZ += az;

                    l_alignementX += vx[another];
                    l_alignementY += vy[another];
                    l_alignementZ += vz[another];

                    float dsqrt = sqrt(d);

                    l_separationX += dx / dsqrt;
                    l_separationY += dy / dsqrt;
                    l_separationZ += dz / dsqrt;

                    l_count += 1;

                }
            }
        }
    }
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
            fvx[tid] = tvx + MAX_ACCELERATION / d * nvx;
            fvy[tid] = tvy + MAX_ACCELERATION / d * nvy;
            fvz[tid] = tvz + MAX_ACCELERATION / d * nvz;
        }
    }
}

__global__ void kernel_move(float *x, float *y, float *z, float *vx, float *vy, float *vz)
{
    unsigned int tid = threadIdx.x + blockIdx.x * blockDim.x;
    float nx = x[tid] + vx[tid];
    float ny = y[tid] + vy[tid];
    float nz = z[tid] + vz[tid];

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
    if(nz < -sea_depth)
        vz[tid] = vz[tid] + TURN_FACTOR;
    else if(nz > sea_depth)
        vz[tid] = vz[tid] - TURN_FACTOR;
    x[tid] = x[tid] + vx[tid];
    y[tid] = y[tid] + vy[tid];
    z[tid] = z[tid] + vz[tid];
}


__global__ void kernel_display(float *pos, float *x, float *y, float *z, float *vx, float *vy, float *vz)
{
    unsigned int tid = threadIdx.x + blockIdx.x * blockDim.x;

    float tvx = vx[tid];
    float tvy = vy[tid];
    float tvz = vz[tid];

    float coef = FISH_LENGTH / sqrt(tvx * tvx + tvy * tvy + tvz * tvz);
    float tempX = coef * tvx;
    float tempY = coef * tvy;
    float tempZ = coef * tvz;

    float p1X = x[tid];
    float p1Y = y[tid];
    float p1Z = z[tid];

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

    initCUDA();
    int targc = 1;
    char** targv = new char*[1];
    targv[0] = argv[0];
    initGL(&targc, targv);

    createVBO(&vbo, &cuda_vbo_resource, cudaGraphicsMapFlagsWriteDiscard);
    glutMainLoop();

    return 0;
}

void initCUDA()
{
    float* h_x = new float[N];
    float* h_y = new float[N];
    float* h_z = new float[N];
    float* h_vx = new float[N];
    float* h_vy = new float[N];
    float* h_vz = new float[N];

    cudaMalloc(&d_constants, sizeof(Constants));
    cudaMemcpy(d_constants, h_constants, sizeof(Constants), cudaMemcpyHostToDevice);
    // srand(time(NULL));
    srand(0);

    for(int i = 0; i < N; ++i)
    {
        h_x[i] = mapRange(0, 100, -sea_width, sea_width, rand() % 100);
        h_y[i] = mapRange(0, 100, -sea_height, sea_height, rand() % 100);
        h_z[i] = mapRange(0, 100, -sea_depth, sea_depth, rand() % 100);

        // std::cout << h_x[i] << " " << h_y[i] << std::endl;

        h_vx[i] = mapRange(0, 100, -MAX_VELOCITY, MAX_VELOCITY, rand() % 100);
        h_vy[i] = mapRange(0, 100, -MAX_VELOCITY, MAX_VELOCITY, rand() % 100);
        h_vz[i] = mapRange(0, 100, -MAX_VELOCITY, MAX_VELOCITY, rand() % 100);
        // printf("Line %d has x: %f y: %f vx: %f vy: %f\n", i, h_x[i], h_y[i], h_vx[i], h_vy[i]);
    }
    cudaMalloc(&d_x, N * sizeof(float));
    cudaMalloc(&d_y, N * sizeof(float));
    cudaMalloc(&d_z, N * sizeof(float));
    cudaMalloc(&d_vx, N * sizeof(float));
    cudaMalloc(&d_vy, N * sizeof(float));
    cudaMalloc(&d_vz, N * sizeof(float));
    cudaMalloc(&d_future_vx, N * sizeof(float));
    cudaMalloc(&d_future_vy, N * sizeof(float));
    cudaMalloc(&d_future_vz, N * sizeof(float));
    cudaMalloc(&d_gridCell, N * sizeof(int));
    cudaMalloc(&d_gridFish, N * sizeof(int));
    cudaMalloc(&d_cellStart, GRID_SIZE * GRID_SIZE * sizeof(int));
    cudaMalloc(&d_startCell, GRID_SIZE * GRID_SIZE * sizeof(int));
    cudaMalloc(&d_endCell, GRID_SIZE * GRID_SIZE * sizeof(int));

    cudaMemcpy(d_x, h_x, N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_y, h_y, N * sizeof(float), cudaMemcpyHostToDevice);    
    cudaMemcpy(d_z, h_z, N * sizeof(float), cudaMemcpyHostToDevice);    
    cudaMemcpy(d_vx, h_vx, N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_vy, h_vy, N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_vz, h_vz, N * sizeof(float), cudaMemcpyHostToDevice);

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
    cudaFree(d_z);
    cudaFree(d_vx);
    cudaFree(d_vy);
    cudaFree(d_vz);
    cudaFree(d_future_vx);
    cudaFree(d_future_vy);
    cudaFree(d_future_vz);
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
    dim3 grid2D(N/1024, 1, 1);
    dim3 block2D(1024, 1, 1);
    dim3 gridGridSize(GRID_SIZE * GRID_SIZE / 1024 + 1, 1, 1);
    dim3 blockGridSize(1024, 1, 1);

    if(doAnimate)
    {
     

        // Prepare helping grid
        setUnsortedGrid<<<grid2D, block2D>>>(d_x, d_y, d_gridCell, d_gridFish);
        thrust::sort_by_key(thrust::device, d_gridCell, d_gridCell + N, d_gridFish);
        cudaMemset(d_cellStart, -1, GRID_SIZE * GRID_SIZE * sizeof(int));
        prepareCellStart<<<grid2D, block2D>>>(d_gridCell, d_cellStart);
        prepareStartEndCell<<<gridGridSize, blockGridSize>>>(d_cellStart, d_startCell, d_endCell, N);

        // Start proper move/animation
        kernel_prepare_move<<<grid2D, block2D>>>(d_x, d_y, d_z, d_vx, d_vy, d_vz, d_future_vx, d_future_vy, d_future_vz, d_gridFish, d_startCell, d_endCell, d_gridCell, d_constants);
        kernel_normalize_velocity<<<grid2D, block2D>>>(d_vx, d_vy, d_vz, d_future_vx, d_future_vy, d_future_vz);
        kernel_move<<<grid2D, block2D>>>(d_x, d_y, d_z, d_vx, d_vy, d_vz);
    }

    kernel_display<<<grid2D, block2D>>>(dptr, d_x, d_y, d_z, d_vx, d_vy, d_vz);

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