#include "fluid_solver.h"
#include <cmath>

#define IX(i, j, k) ((i) + (M + 2) * (j) + (M + 2) * (N + 2) * (k))
#define SWAP(x0, x)                                                            \
  {                                                                            \
    float *tmp = x0;                                                           \
    x0 = x;                                                                    \
    x = tmp;                                                                   \
  }
#define MAX(a, b) (((a) > (b)) ? (a) : (b))
#define MIN(a, b) (((a) < (b)) ? (a) : (b))
#define LINEARSOLVERTIMES 20

// Add sources (density or velocity)
__global__ void add_source_kernel(int size, float *x, float *s, float dt) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        x[idx] += dt * s[idx];
    }
}

void add_source_cuda(int M, int N, int O, float *d_x, float *d_s, float dt) {
    int size = (M + 2) * (N + 2) * (O + 2);

    int threadsPerBlock = 256;
    int blocksPerGrid = (size + threadsPerBlock - 1) / threadsPerBlock;
    add_source_kernel<<<blocksPerGrid, threadsPerBlock>>>(size, d_x, d_s, dt);

    cudaDeviceSynchronize();
}


// Set boundary conditions
__global__
void set_bnd_kernel(int M, int N, int O, int b, float *x)
{

    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    int k = blockIdx.z * blockDim.z + threadIdx.z;

    if (i > (M+1) || j > (N+1) || k > (O+1)) return;

    if (k == 0 && i >= 1 && i <= M && j >= 1 && j <= N) {
        x[IX(i,j,0)] = (b == 3) ? -x[IX(i,j,1)] : x[IX(i,j,1)];
    }

    if (k == (O+1) && i >= 1 && i <= M && j >= 1 && j <= N) {
        x[IX(i,j,O+1)] = (b == 3) ? -x[IX(i,j,O)] : x[IX(i,j,O)];
    }

    if (i == 0 && j >= 1 && j <= N && k >= 1 && k <= O) {
        x[IX(0,j,k)] = (b == 1) ? -x[IX(1,j,k)] : x[IX(1,j,k)];
    }
    if (i == (M+1) && j >= 1 && j <= N && k >= 1 && k <= O) {
        x[IX(M+1,j,k)] = (b == 1) ? -x[IX(M,j,k)] : x[IX(M,j,k)];
    }

    if (j == 0 && i >= 1 && i <= M && k >= 1 && k <= O) {
        x[IX(i,0,k)] = (b == 2) ? -x[IX(i,1,k)] : x[IX(i,1,k)];
    }
    if (j == (N+1) && i >= 1 && i <= M && k >= 1 && k <= O) {
        x[IX(i,N+1,k)] = (b == 2) ? -x[IX(i,N,k)] : x[IX(i,N,k)];
    }

    if (k == 0) {
        if (i == 0 && j == 0) {
            x[IX(0,0,0)] = 0.33f * ( x[IX(1,0,0)]
                                   + x[IX(0,1,0)]
                                   + x[IX(0,0,1)] );
        }
        if (i == (M+1) && j == 0) {
            x[IX(M+1,0,0)] = 0.33f * ( x[IX(M,0,0)]
                                     + x[IX(M+1,1,0)]
                                     + x[IX(M+1,0,1)] );
        }
        if (i == 0 && j == (N+1)) {
            x[IX(0,N+1,0)] = 0.33f * ( x[IX(1,N+1,0)]
                                     + x[IX(0,N,0)]
                                     + x[IX(0,N+1,1)] );
        }
        if (i == (M+1) && j == (N+1)) {
            x[IX(M+1,N+1,0)] = 0.33f * ( x[IX(M,N+1,0)]
                                       + x[IX(M+1,N,0)]
                                       + x[IX(M+1,N+1,1)] );
        }
    }
}

void set_bnd_cuda(int M, int N, int O, int b, float *d_x)
{

    dim3 blockDim(8, 8, 8);
    dim3 gridDim( (M+2 + blockDim.x -1)/blockDim.x,
                  (N+2 + blockDim.y -1)/blockDim.y,
                  (O+2 + blockDim.z -1)/blockDim.z );


    set_bnd_kernel<<< gridDim, blockDim >>>(M, N, O, b, d_x);
    cudaDeviceSynchronize();
}


__device__ static float atomicMaxFloat(float* address, float val)
{
    int* address_as_i = (int*) address;
    int old = *address_as_i, assumed;
    do {
        assumed = old;
        old = ::atomicCAS(address_as_i, assumed,
            __float_as_int(::fmaxf(val, __int_as_float(assumed))));
    } while (assumed != old);
    return __int_as_float(old);
}

// red-black solver with convergence check
__global__
void lin_solve_kernel_red(
    int M, int N, int O,
    float* x, const float* x0,
    float a, float c,
    float* d_max_diff 
)
{

    int i = blockIdx.x * blockDim.x + threadIdx.x + 1;
    int j = blockIdx.y * blockDim.y + threadIdx.y + 1;
    int k = blockIdx.z * blockDim.z + threadIdx.z + 1;

    int localThreadId = threadIdx.x
                      + threadIdx.y * blockDim.x
                      + threadIdx.z * blockDim.x * blockDim.y;

    extern __shared__ float sdata[];

    float diff = 0.0f;

    if (i <= M && j <= N && k <= O) {
        // Update only for "red" cells
        if (((i + j + k) & 1) == 0) {
            int idx = IX(i, j, k);

            float old_x = x[idx];
            float new_x = ( x0[idx]
                            + a * (
                                x[IX(i-1, j,   k  )] + x[IX(i+1, j,   k  )] +
                                x[IX(i,   j-1, k  )] + x[IX(i,   j+1, k  )] +
                                x[IX(i,   j,   k-1)] + x[IX(i,   j,   k+1)]
                            )
                          ) / c;

            x[idx] = new_x;
            diff   = fabsf(new_x - old_x);
        }
    }

    sdata[localThreadId] = diff;
    __syncthreads();

    // Perform a block-level tree reduction to find the max diff in this block
    int totalThreads = blockDim.x * blockDim.y * blockDim.z;
    for (int offset = totalThreads / 2; offset > 0; offset >>= 1) {
        if (localThreadId < offset) {
            sdata[localThreadId] = fmaxf(sdata[localThreadId], sdata[localThreadId + offset]);
        }
        __syncthreads();
    }

    // The first thread in the block updates the global max
    if (localThreadId == 0) {
        atomicMaxFloat(d_max_diff, sdata[0]);
    }
}

__global__
void lin_solve_kernel_black(
    int M, int N, int O,
    float* x, const float* x0,
    float a, float c,
    float* d_max_diff
)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x + 1;
    int j = blockIdx.y * blockDim.y + threadIdx.y + 1;
    int k = blockIdx.z * blockDim.z + threadIdx.z + 1;

    int localThreadId = threadIdx.x
                      + threadIdx.y * blockDim.x
                      + threadIdx.z * blockDim.x * blockDim.y;

    extern __shared__ float sdata[];

    float diff = 0.0f;

    if (i <= M && j <= N && k <= O) {
        // "black" cells
        if (((i + j + k) & 1) == 1) {
            int idx = IX(i, j, k);

            float old_x = x[idx];
            float new_x = ( x0[idx]
                            + a * (
                                x[IX(i-1, j,   k)] + x[IX(i+1, j,   k)] +
                                x[IX(i,   j-1, k)] + x[IX(i,   j+1, k)] +
                                x[IX(i,   j,   k-1)] + x[IX(i,   j,   k+1)]
                            )
                          ) / c;

            x[idx] = new_x;
            diff   = fabsf(new_x - old_x);
        }
    }

    sdata[localThreadId] = diff;
    __syncthreads();

    int totalThreads = blockDim.x * blockDim.y * blockDim.z;
    for (int offset = totalThreads / 2; offset > 0; offset >>= 1) {
        if (localThreadId < offset) {
            sdata[localThreadId] = fmaxf(sdata[localThreadId], sdata[localThreadId + offset]);
        }
        __syncthreads();
    }

    if (localThreadId == 0) {
        atomicMaxFloat(d_max_diff, sdata[0]);
    }
}


void lin_solve_cuda(int M, int N, int O, int b,
                    float *x, float *x0,
                    float a, float c)
{
    // Convergence tolerance
    const float tol = 1e-7f;
    const int maxIter = 20;

    // Device memory to track max change for each iteration
    float *d_max_diff = nullptr;
    cudaMalloc((void**)&d_max_diff, sizeof(float));

    dim3 threadsPerBlock(8, 8, 8);
    dim3 numBlocks( (M + threadsPerBlock.x - 1) / threadsPerBlock.x,
                    (N + threadsPerBlock.y - 1) / threadsPerBlock.y,
                    (O + threadsPerBlock.z - 1) / threadsPerBlock.z );


    size_t smemSize = threadsPerBlock.x * threadsPerBlock.y 
                    * threadsPerBlock.z * sizeof(float);

    int iter = 0;
    while (iter < maxIter) {
        // Reset d_max_diff to 0.0f
        float zero = 0.0f;
        cudaMemcpy(d_max_diff, &zero, sizeof(float), cudaMemcpyHostToDevice);

        // RED 
        lin_solve_kernel_red<<<numBlocks, threadsPerBlock, smemSize>>>(
            M, N, O, x, x0, a, c, d_max_diff
        );
        cudaDeviceSynchronize();

        // BLACK 
        lin_solve_kernel_black<<<numBlocks, threadsPerBlock, smemSize>>>(
            M, N, O, x, x0, a, c, d_max_diff
        );
        cudaDeviceSynchronize();
        set_bnd_cuda(M, N, O, b, x);

        float h_max_diff = 0.0f;
        cudaMemcpy(&h_max_diff, d_max_diff, sizeof(float), cudaMemcpyDeviceToHost);

        iter++;
        if (h_max_diff < tol) {
            break;
        }
    }

    cudaFree(d_max_diff);
}

// Diffusion step (uses implicit method)
void diffuse_cuda(int M, int N, int O, int b, float *x, float *x0, float diff, float dt) {
    int max = MAX(MAX(M, N), O);
    float a = dt * diff * max * max;
    lin_solve_cuda(M, N, O, b, x, x0, a, 1 + 6 * a);
}

// Advection step (uses velocity field to move quantities)
__global__ void advect_kernel(
    int M, int N, int O, int b,
    float *d, const float *d0,
    const float *u, const float *v, const float *w,
    float dt)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x + 1;
    int j = blockIdx.y * blockDim.y + threadIdx.y + 1;
    int k = blockIdx.z * blockDim.z + threadIdx.z + 1;

    if (i > M || j > N || k > O) return;

    float dtX = dt * M;
    float dtY = dt * N;
    float dtZ = dt * O;

    float x = i - dtX * u[IX(i, j, k)];
    float y = j - dtY * v[IX(i, j, k)];
    float z = k - dtZ * w[IX(i, j, k)];

    x = fmaxf(0.5f, fminf(x, M + 0.5f));
    y = fmaxf(0.5f, fminf(y, N + 0.5f));
    z = fmaxf(0.5f, fminf(z, O + 0.5f));

    int i0 = floorf(x);
    int i1 = i0 + 1;
    int j0 = floorf(y);
    int j1 = j0 + 1;
    int k0 = floorf(z);
    int k1 = k0 + 1;

    float s1 = x - i0;
    float s0 = 1.0f - s1;
    float t1 = y - j0;
    float t0 = 1.0f - t1;
    float u1 = z - k0;
    float u0 = 1.0f - u1;

    d[IX(i, j, k)] =
        s0 * (t0 * (u0 * d0[IX(i0, j0, k0)] + u1 * d0[IX(i0, j0, k1)]) +
              t1 * (u0 * d0[IX(i0, j1, k0)] + u1 * d0[IX(i0, j1, k1)])) +
        s1 * (t0 * (u0 * d0[IX(i1, j0, k0)] + u1 * d0[IX(i1, j0, k1)]) +
              t1 * (u0 * d0[IX(i1, j1, k0)] + u1 * d0[IX(i1, j1, k1)]));
}


void advect_cuda(int M, int N, int O, int b, float *d, float *d0,
                float *u, float *v, float *w, float dt){


    dim3 threadsPerBlock(8, 8, 8);
    dim3 blocksPerGrid(
        (M + threadsPerBlock.x - 1) / threadsPerBlock.x,
        (N + threadsPerBlock.y - 1) / threadsPerBlock.y,
        (O + threadsPerBlock.z - 1) / threadsPerBlock.z
    );

    advect_kernel<<<blocksPerGrid, threadsPerBlock>>>(M, N, O, b, d, d0, u, v, w, dt);
    cudaDeviceSynchronize();
    set_bnd_cuda(M, N, O, b, d);
}


// Projection step to ensure incompressibility (make the velocity field
// divergence-free)

__global__ void compute_divergence(
    int M, int N, int O,
    const float *u, const float *v, const float *w,
    float *div, float *p)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x + 1;
    int j = blockIdx.y * blockDim.y + threadIdx.y + 1;
    int k = blockIdx.z * blockDim.z + threadIdx.z + 1;

    if (i > M || j > N || k > O) return;

    div[IX(i, j, k)] = -0.5f * (
        u[IX(i + 1, j, k)] - u[IX(i - 1, j, k)] +
        v[IX(i, j + 1, k)] - v[IX(i, j - 1, k)] +
        w[IX(i, j, k + 1)] - w[IX(i, j, k - 1)]
    ) / MAX(MAX(M, N),O);

    p[IX(i, j, k)] = 0.0f;
}

__global__ void update_velocity(
    int M, int N, int O,
    float *u, float *v, float *w,
    const float *p)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x + 1;
    int j = blockIdx.y * blockDim.y + threadIdx.y + 1;
    int k = blockIdx.z * blockDim.z + threadIdx.z + 1;

    if (i > M || j > N || k > O) return;

    u[IX(i, j, k)] -= 0.5f * (p[IX(i + 1, j, k)] - p[IX(i - 1, j, k)]);
    v[IX(i, j, k)] -= 0.5f * (p[IX(i, j + 1, k)] - p[IX(i, j - 1, k)]);
    w[IX(i, j, k)] -= 0.5f * (p[IX(i, j, k + 1)] - p[IX(i, j, k - 1)]);
}

void project_cuda(int M, int N, int O, float *u, float *v, float *w, float *p, float *div) {
    dim3 threadsPerBlock(8, 8, 8);
    dim3 blocksPerGrid(
        (M + threadsPerBlock.x - 1) / threadsPerBlock.x,
        (N + threadsPerBlock.y - 1) / threadsPerBlock.y,
        (O + threadsPerBlock.z - 1) / threadsPerBlock.z
    );

    compute_divergence<<<blocksPerGrid, threadsPerBlock>>>(M, N, O, u, v, w, div, p);
    cudaDeviceSynchronize();

    set_bnd_cuda(M, N, O, 0, div);
    set_bnd_cuda(M, N, O, 0, p);

    lin_solve_cuda(M, N, O, 0, p, div, 1.0f, 6.0f);

    update_velocity<<<blocksPerGrid, threadsPerBlock>>>(M, N, O, u, v, w, p);

    cudaDeviceSynchronize();

    set_bnd_cuda(M, N, O, 1, u);
    set_bnd_cuda(M, N, O, 2, v);
    set_bnd_cuda(M, N, O, 3, w);
}


// Step function for density
void dens_step_cuda(int M, int N, int O, float *x, float *x0, float *u, float *v, float *w, float diff, float dt) {
    add_source_cuda(M, N, O, x, x0, dt);
    SWAP(x0, x);
    diffuse_cuda(M, N, O, 0, x, x0, diff, dt);
    SWAP(x0, x);
    advect_cuda(M, N, O, 0, x, x0, u, v, w, dt);
}

// Step function for velocity
void vel_step_cuda(int M, int N, int O, float *u, float *v, float *w, float *u0, float *v0, float *w0, float visc, float dt) {
    add_source_cuda(M, N, O, u, u0, dt);
    add_source_cuda(M, N, O, v, v0, dt);
    add_source_cuda(M, N, O, w, w0, dt);
    SWAP(u0, u);
    diffuse_cuda(M, N, O, 1, u, u0, visc, dt);
    SWAP(v0, v);
    diffuse_cuda(M, N, O, 2, v, v0, visc, dt);
    SWAP(w0, w);
    diffuse_cuda(M, N, O, 3, w, w0, visc, dt);
    project_cuda(M, N, O, u, v, w, u0, v0);
    SWAP(u0, u);
    SWAP(v0, v);
    SWAP(w0, w);
    advect_cuda(M, N, O, 1, u, u0, u0, v0, w0, dt);
    advect_cuda(M, N, O, 2, v, v0, u0, v0, w0, dt);
    advect_cuda(M, N, O, 3, w, w0, u0, v0, w0, dt);
    project_cuda(M, N, O, u, v, w, u0, v0);
}
