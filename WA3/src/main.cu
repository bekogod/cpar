// main.cu
#include "EventManager.h"
#include "fluid_solver.h"
#include <iostream>
#include <vector>
#include <cuda_runtime.h>
#include <cmath>

#define SIZE 168
#define IX(i, j, k) ((i) + (M + 2) * (j) + (M + 2) * (N + 2) * (k))

// CUDA Error Checking Macro
#define CUDA_CHECK_ERROR(call)                                           \
    {                                                                    \
        cudaError_t err = call;                                          \
        if (err != cudaSuccess) {                                        \
            std::cerr << "CUDA Error: " << cudaGetErrorString(err)       \
                      << " at " << __FILE__ << ":" << __LINE__ << std::endl; \
            exit(EXIT_FAILURE);                                          \
        }                                                                \
    }

// Globals for the grid size
static int M = SIZE;
static int N = SIZE;
static int O = SIZE;
static float dt = 0.1f;      // Time delta
static float diff = 0.0001f; // Diffusion constant
static float visc = 0.0001f; // Viscosity constant

// Fluid simulation arrays
static float *u, *v, *w, *u_prev, *v_prev, *w_prev;
static float *dens, *dens_prev;

// Function to allocate simulation data
int allocate_data() {
    int size = (M + 2) * (N + 2) * (O + 2);
   
    CUDA_CHECK_ERROR(cudaMalloc((void**)&u,       size * sizeof(float)));
    CUDA_CHECK_ERROR(cudaMalloc((void**)&v,       size * sizeof(float)));
    CUDA_CHECK_ERROR(cudaMalloc((void**)&w,       size * sizeof(float)));
    CUDA_CHECK_ERROR(cudaMalloc((void**)&u_prev,  size * sizeof(float)));
    CUDA_CHECK_ERROR(cudaMalloc((void**)&v_prev,  size * sizeof(float)));
    CUDA_CHECK_ERROR(cudaMalloc((void**)&w_prev,  size * sizeof(float)));
    CUDA_CHECK_ERROR(cudaMalloc((void**)&dens,    size * sizeof(float)));
    CUDA_CHECK_ERROR(cudaMalloc((void**)&dens_prev, size * sizeof(float)));

    return 1;
}

// Function to clear the data (set all to zero)
void clear_data() {
    int size = (M + 2) * (N + 2) * (O + 2);
    
    CUDA_CHECK_ERROR(cudaMemset(u, 0, size * sizeof(float)));
    CUDA_CHECK_ERROR(cudaMemset(v, 0, size * sizeof(float)));
    CUDA_CHECK_ERROR(cudaMemset(w, 0, size * sizeof(float)));
    CUDA_CHECK_ERROR(cudaMemset(u_prev, 0, size * sizeof(float)));
    CUDA_CHECK_ERROR(cudaMemset(v_prev, 0, size * sizeof(float)));
    CUDA_CHECK_ERROR(cudaMemset(w_prev, 0, size * sizeof(float)));
    CUDA_CHECK_ERROR(cudaMemset(dens, 0, size * sizeof(float)));
    CUDA_CHECK_ERROR(cudaMemset(dens_prev, 0, size * sizeof(float)));
    cudaDeviceSynchronize();
}

// Function to free allocated memory
void free_data() {

    CUDA_CHECK_ERROR(cudaFree(u));
    CUDA_CHECK_ERROR(cudaFree(v));
    CUDA_CHECK_ERROR(cudaFree(w));
    CUDA_CHECK_ERROR(cudaFree(u_prev));
    CUDA_CHECK_ERROR(cudaFree(v_prev));
    CUDA_CHECK_ERROR(cudaFree(w_prev));
    CUDA_CHECK_ERROR(cudaFree(dens));
    CUDA_CHECK_ERROR(cudaFree(dens_prev));
}

__global__ void applyds_kernel(int M, int N, int O, float *dens, int i, int j, int k, float density)
{
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        int idx = IX(i, j, k);
        dens[idx] = density;
    }

}

__global__ void applyf_kernel(int M, int N, int O, float *u, float *v, float *w, int i, int j, int k, float fx, float fy, float fz){
    if (threadIdx.x == 0 && blockIdx.x == 0){
        int idx = IX(i, j, k);
        u[idx] = fx;
        v[idx] = fy;
        w[idx] = fz;
    }
}

// Function to apply events (source or force) for the current timestep
void apply_events(const std::vector<Event> &events) {
    for (const auto &event : events) {
        if (event.type == ADD_SOURCE) {
            int i = M / 2, j = N / 2, k = O / 2;
            applyds_kernel<<<1, 1>>>(M, N, O, dens, i, j, k, event.density);
        } else if (event.type == APPLY_FORCE) {
            int i = M / 2, j = N / 2, k = O / 2;
            applyf_kernel<<<1, 1>>>(M, N, O, u, v, w, i, j, k, event.force.x, event.force.y, event.force.z);
        }
    }

    cudaDeviceSynchronize();
}

// Function to sum the total density
float sum_density() {
    float total_density = 0.0f;
    int size = (M + 2) * (N + 2) * (O + 2);

    float *h_dens = new float[size];
    if (!h_dens) {
        std::cerr << "Cannot allocate temporary Host memory for density" << std::endl;
        exit(EXIT_FAILURE);
    }
    
    cudaError_t err = cudaMemcpy(h_dens, dens, size * sizeof(float), cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) {
    std::cerr << "CUDA Error: " << cudaGetErrorString(err) << std::endl;
    }

    for (int i = 0; i < size; i++) {
        total_density += h_dens[i];
    }

    delete[] h_dens;
    return total_density;
}


void check_total_density(int timestep) {
    float total_density = sum_density();
    std::cout << "Total density at timestep " << timestep << ": " << total_density << std::endl;
}

// Simulation loop
void simulate(EventManager &eventManager, int timesteps) {
    for (int t = 0; t < timesteps; t++) {
        // Get the events for the current timestep
        std::vector<Event> events = eventManager.get_events_at_timestamp(t);

        // Apply events to the simulation
        apply_events(events);

        // Perform the simulation steps
        vel_step_cuda(M, N, O, u, v, w, u_prev, v_prev, w_prev, visc, dt);
        dens_step_cuda(M, N, O, dens, dens_prev, u, v, w, diff, dt);

        //check_total_density(t);
    }
}


int main() {
    // Initialize EventManager
    EventManager eventManager;
    eventManager.read_events("events.txt");

    // Get the total number of timesteps from the event file
    int timesteps = eventManager.get_total_timesteps();

    // Allocate and clear data
    if (!allocate_data()) return -1;
    clear_data();

    // Run simulation with events
    simulate(eventManager, timesteps);

    // Print total density at the end of simulation
    float total_density = sum_density();
    std::cout << "Total density after " << timesteps
              << " timesteps: " << total_density << std::endl;

    // Free memory
    free_data();

    return 0;
}
