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
////////////////////////////////////////////////////////////////////////////////
unsigned int N = 1024 * 5;
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

int *gridCell, *gridFish, *cellStart, *startCell, *endCell;

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
Constants *constants;

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

    ~FishData()
    {
        delete[] x;
        delete[] y;
        delete[] z;

        delete[] vx;
        delete[] vy;
        delete[] vz;

        delete[] fvx;
        delete[] fvy;
        delete[] fvz;
    }
};

FishData *fishData;

float *data_vis_ptr;

// vbo variables
GLuint vbo;

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
void createVBO(GLuint *vbo);
void deleteVBO(GLuint *vbo);

void cleanup();
void keyboard(unsigned char key, int x, int y);
void mouse(int button, int state, int x, int y);
void motion(int x, int y);

void computeFPS();
void timerEvent(int value);
void display();
void runAnimation();
void quicksort(int *key, int *value, int first, int last);

void quicksort(int *key, int *value, int first,int last)
{
    int i, j, pivot, temp;

    if(first<last){
        pivot=first;
        i=first;
        j=last;

        while(i<j){
            while(key[i]<=key[pivot]&&i<last)
                i++;
            while(key[j]>key[pivot])
                j--;
            if(i<j){
                temp=key[i];
                key[i]=key[j];
                key[j]=temp;
                temp=value[i];
                value[i]=value[j];
                value[j]=temp;
            }
        }

        temp=key[pivot];
        key[pivot]=key[j];
        key[j]=temp;
        temp=value[pivot];
        value[pivot]=value[j];
        value[j]=temp;
        quicksort(key, value, first,j-1);
        quicksort(key, value, j+1,last);

   }
}

void setUnsortedGrid(FishData *data, int* gridCell, int* gridFish)
{
    for(int i = 0; i < N; i++)
    {
        int column = (data->x[i] + sea_width) / cell_width;
        int row = (data->y[i] + sea_height) / cell_height;
        gridCell[i] = max(min(row * GRID_SIZE + column, GRID_SIZE * GRID_SIZE - 1), 0);
        gridFish[i] = i;
    }
}

void prepareCellStart(int* gridCell, int* cellStart)
{
    cellStart[gridCell[0]] = 0;
    for(int i = 1; i < N; i++)
    {
        if(gridCell[i] != gridCell[i - 1])
            cellStart[gridCell[i]] = i;
    }
}


void normalize_velocity(FishData *data)
{
    for(int i = 0; i < N; i++)
    {
        float tvx = data->fvx[i];
        float tvy = data->fvy[i];
        float tvz = data->fvz[i];
        
        float speed = sqrt(tvx * tvx + tvy * tvy + tvz * tvz);
        if(speed == 0)
            continue;
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
        data->vx[i] = tvx;
        data->vy[i] = tvy;
        data->vz[i] = tvz;
    }
}

void prepareStartEndCell(int* cellStart, int* startCell, int* endCell)
{
    for(int i = 0; i < GRID_SIZE * GRID_SIZE; i++)
    {
        int startPos = i - GRID_RANGE;
        int endPos = i + GRID_RANGE + 1;

        int tidRow = i / GRID_SIZE;
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

        startCell[i] = startPos == GRID_SIZE * GRID_SIZE ? N : cellStart[startPos];
        endCell[i] = endPos == GRID_SIZE * GRID_SIZE ? N : cellStart[endPos];
    }
}

void prepare_move(FishData *data,
        int* gridFish, int* startCell, int* endCell, int* gridCell,
        struct Constants *consts)
{
    for(int j = 0; j < N; j++)
    {

        int cell = gridCell[j];
        int i = gridFish[j];

        // prepare accumulators
        float l_cohesionX = 0, l_cohesionY = 0, l_cohesionZ = 0;
        float l_alignementX = 0, l_alignementY = 0, l_alignementZ = 0;
        float l_separationX = 0, l_separationY = 0, l_separationZ = 0;
        float l_count = 0;

        // prepare variables
        float tvx = data->vx[i];
        float tvy = data->vy[i];
        float tvz = data->vz[i];

        float tx = data->x[i];
        float ty = data->y[i];
        float tz = data->z[i];

        for(int j = -GRID_RANGE; j <= GRID_RANGE; j++)
        {
            int newCell = cell + j * GRID_SIZE;
            if(newCell >= 0 && newCell < GRID_SIZE * GRID_SIZE)
            {
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
                data->fvx[i] = tvx + MAX_ACCELERATION / d * nvx;
                data->fvy[i] = tvy + MAX_ACCELERATION / d * nvy;
                data->fvz[i] = tvz + MAX_ACCELERATION / d * nvz;
            }
        }
    }
}

void move(FishData *data)
{
    for(int i = 0; i < N; i++)
    {
        float nx = data->x[i] + data->vx[i];
        float ny = data->y[i] + data->vy[i];
        float nz = data->z[i] + data->vz[i];

        // repair X velocity
        if(nx < -sea_width)
            data->vx[i] = data->vx[i] + TURN_FACTOR;
        else if(nx > sea_width)
            data->vx[i] = data->vx[i] - TURN_FACTOR;
        // repair Y velocity
        if(ny < -sea_height)
            data->vy[i] = data->vy[i] + TURN_FACTOR;
        else if(ny > sea_height)
            data->vy[i] = data->vy[i] - TURN_FACTOR;
        // repair Z velocity
        if(nz < -sea_depth)
            data->vz[i] = data->vz[i] + TURN_FACTOR;
        else if(nz > sea_depth)
            data->vz[i] = data->vz[i] - TURN_FACTOR;
        data->x[i] = data->x[i] + data->vx[i];
        data->y[i] = data->y[i] + data->vy[i];
        data->z[i] = data->z[i] + data->vz[i];
    }
}


void triangle_display(float *pos, FishData *data)
{
    for(int i = 0; i < N; i++)
    {
        float tvx = data->vx[i];
        float tvy = data->vy[i];
        float tvz = data->vz[i];

        float coef = FISH_LENGTH / sqrt(tvx * tvx + tvy * tvy + tvz * tvz);
        float tempX = coef * tvx;
        float tempY = coef * tvy;
        float tempZ = coef * tvz;

        float p1X = data->x[i];
        float p1Y = data->y[i];
        float p1Z = data->z[i];

        float p2X = p1X;
        float p2Y = p1Y + tempZ * FISH_WIDTH / FISH_LENGTH;
        float p2Z = p1Z - tempY * FISH_WIDTH / FISH_LENGTH;

        float p3X = p1X;
        float p3Y = p1Y - tempZ * FISH_WIDTH / FISH_LENGTH;
        float p3Z = p1Z + tempY * FISH_WIDTH / FISH_LENGTH;

        p1X += tempX;
        p1Y += tempY;
        p1Z += tempZ;

        pos[9 * i] = p1X / window_width;
        pos[9 * i + 1] = p1Y / window_height;
        pos[9 * i + 2] = p1Z / window_depth;
        pos[9 * i + 3] = p2X / window_width;
        pos[9 * i + 4] = p2Y / window_height;
        pos[9 * i + 5] = p2Z / window_depth;
        pos[9 * i + 6] = p3X / window_width;
        pos[9 * i + 7] = p3Y / window_height;
        pos[9 * i + 8] = p3Z / window_depth;
    }
}


int main(int argc, char **argv)
{
    constants = new Constants(4.0f, 4.0f, 4.0f);
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

    createVBO(&vbo);
    glutMainLoop();

    return 0;
}

void initMemory()
{

    float* h_x = new float[N];
    float* h_y = new float[N];
    float* h_z = new float[N];
    float* h_vx = new float[N];
    float* h_vy = new float[N];
    float* h_vz = new float[N];
    float* h_fvx = new float[N];
    float* h_fvy = new float[N];
    float* h_fvz = new float[N];
    data_vis_ptr = new float[9 * N];

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

    fishData = new FishData();
    fishData->x = h_x;
    fishData->y = h_y;
    fishData->z = h_z;

    fishData->vx = h_vx;
    fishData->vy = h_vy;
    fishData->vz = h_vz;

    fishData->fvx = h_fvx;
    fishData->fvy = h_fvy;
    fishData->fvz = h_fvz;

    gridCell = new int[N];
    gridFish = new int[N];
    cellStart = new int[GRID_SIZE * GRID_SIZE];
    startCell = new int[GRID_SIZE * GRID_SIZE];
    endCell = new int[GRID_SIZE * GRID_SIZE];
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
        deleteVBO(&vbo);
    }
    delete fishData;
    delete constants;
}

void createVBO(GLuint *vbo)
{
    assert(vbo);

    // create buffer object
    glGenBuffers(1, vbo);
    glBindBuffer(GL_ARRAY_BUFFER, *vbo);

    SDK_CHECK_ERROR_GL();
}

void deleteVBO(GLuint *vbo)
{
    glBindBuffer(1, *vbo);
    glDeleteBuffers(1, vbo);

    *vbo = 0;
}

void display()
{
    sdkStartTimer(&timer);

    runAnimation();
    triangle_display(data_vis_ptr, fishData);

    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    // set view matrix
    glMatrixMode(GL_MODELVIEW);
    glLoadIdentity();
    glTranslatef(0.0, 0.0, translate_z);
    glRotatef(rotate_x, 1.0, 0.0, 0.0);
    glRotatef(rotate_y, 0.0, 1.0, 0.0);

    // render from the vbo
    glBindBuffer(GL_ARRAY_BUFFER, vbo);
    glBufferData(GL_ARRAY_BUFFER, 9 * N * sizeof(float), data_vis_ptr, GL_DYNAMIC_DRAW);
    glVertexPointer(3, GL_FLOAT, 0, 0);

    glEnableClientState(GL_VERTEX_ARRAY);
    glColor3f(1.0, 0.0, 0.0);
    glDrawArrays(GL_TRIANGLES, 0, 9 * N);
    glDisableClientState(GL_VERTEX_ARRAY);

    glutSwapBuffers();
    sdkStopTimer(&timer);
    computeFPS();
}

void runAnimation()
{
    if(doAnimate)
    {
        // Prepare helping grid
        setUnsortedGrid(fishData, gridCell, gridFish);
        quicksort(gridCell, gridFish, 0, N - 1);
        for(int i = 0; i < GRID_SIZE * GRID_SIZE; i++)
        {
            cellStart[i] = -1;
        }
        prepareCellStart(gridCell, cellStart);
        prepareStartEndCell(cellStart, startCell, endCell);

        // Start proper move/animation
        prepare_move(fishData, gridFish, startCell, endCell, gridCell, constants);
        normalize_velocity(fishData);
        move(fishData);
    }
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
        case ('d'):
            for(int i = 0; i < N; i++)
            {
                std::cout << "x: " << fishData->fvx[i] << " y: " << fishData->fvy[i] << " z: " << fishData->fvz[i] << std::endl;
            }
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