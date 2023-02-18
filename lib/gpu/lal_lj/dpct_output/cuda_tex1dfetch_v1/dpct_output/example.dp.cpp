#include <CL/sycl.hpp>
#include <dpct/dpct.hpp>
#include <stdio.h>
#define NUM_FLOATS 16

/*
DPCT1059:2: SYCL only supports 4-channel image format. Adjust the code.
*/
dpct::image_wrapper<float, 1> tex1;

void
TexReadout( float *out, size_t N , sycl::nd_item<3> item_ct1,
            dpct::image_accessor_ext<float, 1> tex1)
{
    for (size_t i = item_ct1.get_group(2) * item_ct1.get_local_range().get(2) +
                    item_ct1.get_local_id(2);
         i < N;
         i += item_ct1.get_group_range(2) * item_ct1.get_local_range().get(2))
    {
        out[i] = tex1.read(i);
    }
}

int
main( int argc, char *argv[] )
{
    dpct::device_ext &dev_ct1 = dpct::get_current_device();
    sycl::queue &q_ct1 = dev_ct1.default_queue();
    int ret = 1;
    float *p = 0;
    float *finHost;
    float *finDevice;

    float *foutHost;
    float *foutDevice;
    int status;
    dpct::device_info props;

    /*
    DPCT1027:3: The call to cudaSetDeviceFlags was replaced with 0 because DPC++
    currently does not support setting flags for devices.
    */
    cuda(0);
    /*
    DPCT1003:4: Migrated API does not return error code. (*, 0) is inserted. You
    may need to rewrite this code.
    */
    cuda((dpct::dev_mgr::instance().get_device(0).get_device_info(props), 0));
    cuda(Malloc( (void **) &p, NUM_FLOATS*sizeof(float)) );
    /*
    DPCT1048:0: The original value cudaHostAllocMapped is not meaningful in the
    migrated code and was removed or replaced with 0. You may need to check the
    migrated code.
    */
    cuda(HostAlloc((void **)&finHost, NUM_FLOATS * sizeof(float), 0));
    /*
    DPCT1003:5: Migrated API does not return error code. (*, 0) is inserted. You
    may need to rewrite this code.
    */
    cuda((*((void **)&finDevice) = finHost, 0));

    /*
    DPCT1048:1: The original value cudaHostAllocMapped is not meaningful in the
    migrated code and was removed or replaced with 0. You may need to check the
    migrated code.
    */
    cuda(HostAlloc((void **)&foutHost, NUM_FLOATS * sizeof(float), 0));
    /*
    DPCT1003:6: Migrated API does not return error code. (*, 0) is inserted. You
    may need to rewrite this code.
    */
    cuda((*((void **)&foutDevice) = foutHost, 0));

    for ( int i = 0; i < NUM_FLOATS; i++ ) {
        finHost[i] = (float) i;
    }

    {
        size_t offset;
        cuda(BindTexture( &offset, tex1, finDevice, NUM_FLOATS*sizeof(float)) );
    }

    ret = 0;
Error:
    /*
    DPCT1003:7: Migrated API does not return error code. (*, 0) is inserted. You
    may need to rewrite this code.
    */
    (sycl::free(p, q_ct1), 0);
    sycl::free(finHost, q_ct1);
    sycl::free(foutHost, q_ct1);
    return ret;
}
