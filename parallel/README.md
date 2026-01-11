# Program Purpose
- Demonstrates GPU parallel processing using NVIDIA CUDA by computing the dot product of 2 large vectors. 
- Measures performance using ctime’s clock() and CUDA events. 

# Thread and Block Mapping

	int tid = threadIdx.x + blockIdx.x * blockDim.x; 
 
- Each thread computes partial dot products. 
- Threads cover the entire input using grid-stride looping: 

	double temp = 0; 
	while (tid < len) {
  		temp += a[tid] * b[tid];
    	tid += blockDim.x * gridDim.x;
  	}

# Shared Memory Reduction

	__shared__ double sums[blockDim.x]; 

- Each thread stores its partial sum in shared memory local to its block. 
- Shared memory is much faster than global memory.

### Tree-Based Parallel Reduction Method: 
  	while (halfLen != 0) {
    	if (sumsIdx < halfLen) {
        	sums[sumsIdx] += sums[sumsIdx + halfLen];
    	}
    	halfLen /= 2;
    	__syncthreads();
  	}

- Time complexity: O(log blockDim.x)
- __synthreads() to ensure data isn’t pulled before it’s properly updated.

	if (sumsIdx == 0) {
		res[blockIdx.x] = sums[0];
  	}

- Each block’s shared memory’s array’s 0th element gets the block’s partial dot product.
- These values are passed to the host for the CPU to do the final summation

# Performance Measuring
- Uses ctime’s clock() method to measure a purely CPU dot product function’s time.
- Uses CUDA Events to measure the GPU parallel processing kernel function described above.
- Each function is run 1000 times for stable timing measurement.

# Memory Management
- Free the vectors’ dynamically allocated memory from both the host and device. 
