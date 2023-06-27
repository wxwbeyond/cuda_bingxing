#include<iostream>
#include<fstream>
#include<stdio.h>
#include<stdlib.h>
#include <iomanip>
#include <Windows.h>
#include"cuda_runtime.h"
#include"device_launch_parameters.h"
#include<immintrin.h>  // AVX

using namespace std;
const int N = 2500;
const int BLOCK_SIZE = 1024;
float elm[N][N] = { 0 };

//消去算法
void C_GE(float** a, int n) // 高斯消去算法的Cache优化版本
{  
	float t1, t2;  
	for (int k = 0; k < n; k++)
	{
		t1 = a[k][k];
		for (int j = k + 1; j < n; j++)
			a[k][j] = a[k][j] / t1;
		a[k][k] = 1.0;
		for (int i = k + 1; i < n; i++)
		{
			t2 = a[i][k];
			for (int j = k + 1; j < n; j++)
				a[i][j] -= t2 * a[k][j];
			a[i][k] = 0;
		}
	}
}


//CUDA算法核函数
__global__ void div_kernel(float* data, int k, int N) 
{
	//int tid = blockDim.x * blockIdx.x + threadIdx.x;  //计算线程索引，这里线程格大小为（1，1），所以不需要计算gridDim有关信息
	int tid = threadIdx.x;  //计算线程索引，这里线程格大小为（1，1），所以不需要计算gridDim有关信息
	float element = data[k * N + k];
	while (k + tid + 1 < N) 
	{
		data[k * (N + 1) + tid + 1] /= element;
		tid += blockDim.x;
	}
	return;
}

__global__ void elim_kernel(float* data, int k, int N)
{
	int tx = blockDim.x * blockIdx.x + threadIdx.x;
	if (!tx) data[k * N + k] = 1.0;//对角线元素设为 1
	int row = k + 1 + blockIdx.x;//每个块负责一行
	float t;
	while (row < N)
	{
		int tid = threadIdx.x;
		t = data[(row * N) + k];
		while (k + 1 + tid < N) 
		{
			int col = k + 1 + tid;
			data[(row * N) + col] = data[(row * N) + col] - t * data[k * N + col];
			tid = tid + blockDim.x;
		}
		__syncthreads();
		//块内同步
		if (threadIdx.x == 0) data[row * N + k] = 0;
		row += gridDim.x;
	}
	return;
}

//展示所有内容
float** generate(int n)
{
	ifstream inp("input.txt"); 
	inp >> n;
	float** m = new float* [n];
	for (int i = 0; i < n; i++)
	{
		m[i] = new float[n];
		for (int j = 0; j < n; j++) inp >> m[i][j];
	}
	inp.close();
	return m;
}

float* generate_1d(int n) 
{
	ifstream inp("input.txt"); inp >> n;
	float* m = new float[n * n];
	for (int i = 0; i < n; i++) 
	for (int j = 0; j < n; j++) 
	inp >> m[i * n + j];
	inp.close();
	return m;
}

//CUDA消去算法
void CUDA_GE(float* in)
{
	//show_1d(temp, N);
	cudaError_t ret;//用于错误检查，当 CUDA 接口调用成功会返回 cudaSucess
	float* gpudata;
	float* result = new float[N * N];
	int size = N * N * sizeof(float);

	//分配显存空间并且进行错误检查
	if (cudaMalloc(&gpudata, size) != cudaSuccess)  printf("cudaMalloc gpudata failed!\n");
	//将数据传输至 GPU 端并进行错误检查
	if (cudaMemcpy(gpudata, in, size, cudaMemcpyHostToDevice) != cudaSuccess) printf("cudaMemcpyHostToDevice failed!\n");

	dim3 dimBlock(BLOCK_SIZE, 1), dimGrid(1, 1); //线程块、线程网格

	cudaEvent_t start, stop;  //计时器
	float elapsedTime = 0.0;
	cudaEventCreate(&start), cudaEventCreate(&stop);
	cudaEventRecord(start, 0);  //开始计时

	cudaError_t exec;
	for (int k = 0; k < N; k++) 
	{

		div_kernel << <dimGrid, dimBlock >> > (gpudata, k, N);//负责除法任务的核函数

		cudaDeviceSynchronize();//CPU 与 GPU 之间的同步函数
		exec = cudaGetLastError();
		if (exec != cudaSuccess) printf("division_kernel failed, %s\n", cudaGetErrorString(exec));

		elim_kernel << <dimGrid, dimBlock >> > (gpudata, k, N);//负责消去任务的核函数

		cudaDeviceSynchronize();
		exec = cudaGetLastError();
		if (exec != cudaSuccess) printf("eliminate_kernel failed, %s\n", cudaGetErrorString(exec));
	}

	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);//停止计时
	cudaEventElapsedTime(&elapsedTime, start, stop);
	printf("CUDA_GE:%f ms\n", elapsedTime);

	cudaError_t cudaStatus2 = cudaGetLastError();
	if (cudaGetLastError() != cudaSuccess) fprintf(stderr, "Kernel launch failed: %s\n", cudaGetErrorString(cudaStatus2));
	//将数据传回 CPU 端并进行错误检查
	if (cudaMemcpy(result, gpudata, size, cudaMemcpyDeviceToHost) != cudaSuccess) printf("cudaMemcpyDeviceToHost failed!\n");


	cudaFree(gpudata);//释放显存空间，用 CUDA 接口分配的空间必须用 cudaFree 释放

	//销毁计时器
	cudaEventDestroy(start);
	cudaEventDestroy(stop);
}

void CUDA_GE_Opt(float* in) 
{
	//show_1d(temp, N);
	cudaError_t ret;//用于错误检查，当 CUDA 接口调用成功会返回 cudaSucess
	float* gpudata;
	float* result = new float[N * N];
	int size = N * N * sizeof(float);

	//分配显存空间并且进行错误检查
	if (cudaMallocManaged(&gpudata, size) != cudaSuccess)  printf("cudaMalloc gpudata failed!\n");
	//将数据传输至 GPU 端并进行错误检查
	if (cudaMemcpy(gpudata, in, size, cudaMemcpyHostToDevice) != cudaSuccess) printf("cudaMemcpyHostToDevice failed!\n");

	int deviceId;
	int numberOfSMs;
	int my_cudaDevAttrConcurrentManagedAccess;

	cudaGetDevice(&deviceId);
	cudaDeviceGetAttribute(&numberOfSMs, cudaDevAttrMultiProcessorCount, deviceId);

	int number_of_blocks = numberOfSMs;  // 格大小
	int threads_per_block = 1024;  // 块大小

	//cudaMemPrefetchAsync(gpudata, size, deviceId);  // 重要，但是本机不支持，所以用不了了

	cudaEvent_t start, stop;  //计时器
	float elapsedTime = 0.0;
	cudaEventCreate(&start), cudaEventCreate(&stop);
	cudaEventRecord(start, 0);  //开始计时

	cudaError_t exec;
	for (int k = 0; k < N; k++) {

		div_kernel << <number_of_blocks, threads_per_block >> > (gpudata, k, N);//负责除法任务的核函数

		cudaDeviceSynchronize();//CPU 与 GPU 之间的同步函数
		exec = cudaGetLastError();
		if (exec != cudaSuccess) printf("division_kernel failed, %s\n", cudaGetErrorString(exec));

		elim_kernel << <number_of_blocks, threads_per_block >> > (gpudata, k, N);//负责消去任务的核函数

		cudaDeviceSynchronize();
		exec = cudaGetLastError();
		if (exec != cudaSuccess) printf("eliminate_kernel failed, %s\n", cudaGetErrorString(exec));
	}

	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);//停止计时
	cudaEventElapsedTime(&elapsedTime, start, stop);
	printf("CUDA_GE_Opt:%f ms\n", elapsedTime);

	cudaError_t cudaStatus2 = cudaGetLastError();
	if (cudaGetLastError() != cudaSuccess) fprintf(stderr, "Kernel launch failed: %s\n", cudaGetErrorString(cudaStatus2));
	//将数据传回 CPU 端并进行错误检查
	if (cudaMemcpy(result, gpudata, size, cudaMemcpyDeviceToHost) != cudaSuccess) printf("cudaMemcpyDeviceToHost failed!\n");

	//show_1d(result, N);  // 测试
	show_in_file_1d(result, N);  // 测试

	cudaFree(gpudata);//释放显存空间，用 CUDA 接口分配的空间必须用 cudaFree 释放

	//销毁计时器
	cudaEventDestroy(start);
	cudaEventDestroy(stop);
}


int main()
{
	//freopen("input.txt", "r", stdin);
	//float* temp = new float[N * N];
	//for (int i = 0; i < N; i++) for (int j = 0; j < N; j++) cin >> temp[i * N + j];

	cout << "问题规模为" << N << "，使用固定初始值" << endl;

	long long head, tail, freq;
	QueryPerformanceFrequency((LARGE_INTEGER*)&freq);
	//-----------------------------------------------------------------
	//float** m1 = new float* [n];
	float** m1 = generate(N);
	//reset(m1, n);

	QueryPerformanceCounter((LARGE_INTEGER*)&head);
	C_GE(m1, N);
	QueryPerformanceCounter((LARGE_INTEGER*)&tail);

	cout << "C_GE: " << (tail - head) * 1000.0 / freq
		<< "ms" << endl;

	//-----------------------------------------------------------------
	float* t_1d = generate_1d(N);
	CUDA_GE(t_1d);

	//-----------------------------------------------------------------
	t_1d = generate_1d(N);
	CUDA_GE_Opt(t_1d);

	//system("pause");
}
