#include <stdio.h>
#include <cuda.h>

#define N 32
#define M 128

// texture object is a kernel argument
__global__ void kernel(cudaTextureObject_t tex) {
  int i = blockIdx.x *blockDim.x + threadIdx.x;
  float x = tex1Dfetch<float>(tex, i);
  if (i < 256) printf("%d %f\n", i, x);
}

void call_kernel(cudaTextureObject_t tex) {
  dim3 block(128,1,1);
  dim3 grid((N*M)/block.x,1,1);
  kernel <<<grid, block>>>(tex);
}

int main() {
  // declare and allocate memory
  float *buffer, *h_buffer;
  size_t pitch;
  cudaMallocPitch(&buffer, &pitch, N*sizeof(float),M);
  printf("pitch = %lu\n", pitch);
  cudaMemset(buffer, 0, M*pitch);
  h_buffer=(float *)malloc(N*M*sizeof(float));
  for (int i = 0; i < M; i++)
    for (int j = 0; j < N; j++) h_buffer[i*N+j] = i+1;
  cudaMemcpy2D(buffer, pitch, h_buffer, N*sizeof(float), N*sizeof(float), M,  cudaMemcpyHostToDevice);
  // create texture object
  cudaResourceDesc resDesc;
  memset(&resDesc, 0, sizeof(resDesc));
  resDesc.resType = cudaResourceTypeLinear;
  resDesc.res.linear.devPtr = buffer;
  resDesc.res.linear.desc.f = cudaChannelFormatKindFloat;
  resDesc.res.linear.desc.x = 32; // bits per channel
  resDesc.res.linear.sizeInBytes = M*pitch*sizeof(float);

  cudaTextureDesc texDesc;
  memset(&texDesc, 0, sizeof(texDesc));
  texDesc.readMode = cudaReadModeElementType;

  // create texture object: we only have to do this once!
  cudaTextureObject_t tex=0;
  cudaCreateTextureObject(&tex, &resDesc, &texDesc, NULL);

  call_kernel(tex); // pass texture as argument

  // destroy texture object
  cudaDestroyTextureObject(tex);

  cudaFree(buffer);
}