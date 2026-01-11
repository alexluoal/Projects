#include <cuda_runtime.h>
#include <ctime>
#include <iostream>

using namespace std; 

const int threadsPerBlock = 256; 

__global__ void dotProd( long long len, double* a, double* b, double* res) {

	__shared__ double sums[threadsPerBlock]; 


	int tid = threadIdx.x + blockIdx.x * blockDim.x; 
	int sumsIdx = threadIdx.x; 

	double temp = 0; 

	while (tid < len) {
		temp += a[tid] * b[tid]; 
		tid += blockDim.x * gridDim.x; 
	}

	sums[sumsIdx] = temp; 
	__syncthreads(); 


	int halfLen = blockDim.x / 2; 
	
	while (halfLen != 0) {
		if (sumsIdx < halfLen) {
			sums[sumsIdx] += sums[sumsIdx + halfLen]; 
		}
		halfLen /= 2; 
		__syncthreads(); 	
	}

	if (sumsIdx == 0) {
		res[blockIdx.x] = sums[0]; 
	}
}

double cpu_dotProd( long long len, double* a, double* b) {
	double sum = 0; 
	for (int i = 0; i < len; i++) {
		sum += a[i] * b[i]; 
	}
	return sum; 
}

int main() {
	double cpu_ans = 0; 

	long long len; 
	double *h_a, *h_b, *h_res; 
	double *d_a, *d_b, *d_res; 	

	cout << "Input length: "; 
	cin >> len;
	const int blocksPerGrid = min( (long long)32, (len + threadsPerBlock-1) / threadsPerBlock); 
	h_res = new double[blocksPerGrid]; 

	h_a = new double[len]; 
	h_b = new double[len]; 	
	for (int i = 0; i < len; i++) {
		h_a[i] = i / 2.0; 
		h_b[i] = i / 3.0;
	}

	clock_t cstart = clock(); 
	for (int i = 0; i < 1000; i++) {
		cpu_ans = cpu_dotProd( len, h_a, h_b);  
	}
	clock_t cend = clock(); 

	double time_sec = double(cend - cstart) / CLOCKS_PER_SEC; 
	cout << "CPU sum = " << cpu_ans << ", CPU time = " << time_sec * 1000 << " milliseconds" << endl; 

	cudaMalloc( &d_a, len * sizeof(double) ); 
	cudaMalloc( &d_b, len * sizeof(double) );  
	cudaMalloc( &d_res, blocksPerGrid * sizeof(double) ); 

	cudaMemcpy( d_a, h_a, len * sizeof(double), cudaMemcpyHostToDevice ); 
	cudaMemcpy( d_b, h_b, len * sizeof(double), cudaMemcpyHostToDevice ); 

	cudaEvent_t start, stop; 
	cudaEventCreate(&start); 
	cudaEventCreate(&stop); 

	cudaEventRecord(start);
	for (int i = 0; i < 1000; i++) {		
		dotProd<<<blocksPerGrid, threadsPerBlock>>>( len, d_a, d_b, d_res ); 
	}
	cudaEventRecord(stop); 
	cudaEventSynchronize(stop); 

	float ms; 
	cudaEventElapsedTime( &ms, start, stop); 	

	cudaMemcpy( h_res, d_res, blocksPerGrid * sizeof(double), cudaMemcpyDeviceToHost ); 

	double gpu_ans = 0; 
	for (int i = 0; i < blocksPerGrid; i++) {
		gpu_ans += h_res[i]; 
	}

	cout << "GPU sum = " << gpu_ans << ", GPU time = " << ms << " milliseconds" << endl; 

	delete [] h_a; 
	delete [] h_b; 
	delete [] h_res; 

	cudaFree(d_a); 
	cudaFree(d_b);
	cudaFree(d_res); 	
}
