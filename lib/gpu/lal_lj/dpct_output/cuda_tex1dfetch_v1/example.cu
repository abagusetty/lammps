#include <stdio.h>
#define NUM_FLOATS 16

texture<float, 1> tex1;

__global__ void
TexReadout( float *out, size_t N )
{
    for ( size_t i = blockIdx.x*blockDim.x + threadIdx.x; 
          i < N; 
          i += gridDim.x*blockDim.x )   
    {
        out[i] = tex1Dfetch( tex1, i );
    }
}

int
main( int argc, char *argv[] )
{
    int ret = 1;
    float *p = 0;
    float *finHost;
    float *finDevice;

    float *foutHost;
    float *foutDevice;
    cudaError_t status;
    cudaDeviceProp props;

    // cuda(SetDeviceFlags(cudaDeviceMapHost));
    // cuda(GetDeviceProperties( &props, 0));
    // cuda(Malloc( (void **) &p, NUM_FLOATS*sizeof(float)) );
    // cuda(HostAlloc( (void **) &finHost, NUM_FLOATS*sizeof(float), cudaHostAllocMapped));
    // cuda(HostGetDevicePointer( (void **) &finDevice, finHost, 0 ));

    // cuda(HostAlloc( (void **) &foutHost, NUM_FLOATS*sizeof(float), cudaHostAllocMapped));
    // cuda(HostGetDevicePointer( (void **) &foutDevice, foutHost, 0 ));

    // for ( int i = 0; i < NUM_FLOATS; i++ ) {
    //     finHost[i] = (float) i;
    // }

    {
        size_t offset;
        cuda(BindTexture( &offset, tex1, finDevice, NUM_FLOATS*sizeof(float)) );
    }

    ret = 0;

    cudaFree( p );
    cudaFreeHost( finHost );
    cudaFreeHost( foutHost );
    return ret;
}
