/*
 * Copyright 2015, 2016 (C) Inwoo Chung (gutomitai@gmail.com)
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * 		http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

/*
 * Title: Matrix native functions for CUDA.
 * Author: Inwoo Chung (gutomitai@gmail.com)
 * Since: Jan. 10, 2017
 */

#include <jni.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include "maum_dm_Matrix.h"

// Kernel definition.
__global__ void multiVecs(jdouble *v1, jdouble *v2, jdouble *r, jdouble *sum, jsize length) {
	int i = blockDim.x * blockIdx.x + threadIdx.x;

	if (i < length)
		*sum += v1[i] * v2[i];
}

JNIEXPORT jdouble JNICALL Java_maum_dm_Matrix_innerProductCUDA
  (JNIEnv *env, jclass thisClass, jdoubleArray jV1, jdoubleArray jV2) {

	// Convert JNI double array vectors into native double array vectors.
	jsize length = env->GetArrayLength(jV1);
	size_t size = length * sizeof(jdouble);

	jdouble *v1 = env->GetDoubleArrayElements(jV1, NULL);
	jdouble *v2 = env->GetDoubleArrayElements(jV2, NULL);
	jdouble *r = (jdouble *)malloc(size);

	// Allocate vectors in device memory.
	jdouble *dV1, *dV2, *dR;

	cudaMalloc(&dV1, size); // Why double pointer?
	cudaMalloc(&dV2, size);
	cudaMalloc(&dR, size);

	// Copy vectors from host memory to device memory.
	cudaMemcpy(dV1, v1, size, cudaMemcpyHostToDevice);
	cudaMemcpy(dV2, v2, size, cudaMemcpyHostToDevice);

	// Invoke kernel.
	int threadsPerBlock = 256;
	int blocksPerGrid = (length + threadsPerBlock - 1) / threadsPerBlock;

	// Multiply each element of two vectors.
	jdouble sum;

	multiVecs<<<blocksPerGrid, threadsPerBlock>>>(dV1, dV2, dR, &sum, length);

	// Copy result from device memory to host memory.
	cudaMemcpy(r, dR, size, cudaMemcpyDeviceToHost);

	// Free device memory.
	cudaFree(dV1);
	cudaFree(dV2);
	cudaFree(dR);

	// Free host memory.
	free(v1);
	free(v2);
	free(r);

	return sum;
}
