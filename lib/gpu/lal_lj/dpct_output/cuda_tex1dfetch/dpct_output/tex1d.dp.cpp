#include <CL/sycl.hpp>
#include <dpct/dpct.hpp>
#include <stdio.h>

#define N 32
#define M 128

// texture object is a kernel argument
void kernel(dpct::image_accessor_ext<float, 1> tex,
	    sycl::nd_item<3> item_ct1) {
  int i = item_ct1.get_group(2) * item_ct1.get_local_range().get(2) +
          item_ct1.get_local_id(2);
  float x = tex.read(i);
}

void call_kernel(dpct::image_wrapper_base_p tex) {
  sycl::range<3> block(1, 1, 128);
  sycl::range<3> grid(1, 1, (N * M) / block[2]);
  /*
  DPCT1049:1: The workgroup size passed to the SYCL kernel may exceed the limit.
  To get the device limit, query info::device::max_work_group_size. Adjust the
  workgroup size if needed.
  */
  dpct::get_default_queue().submit([&](sycl::handler &cgh) {
    sycl::stream stream_ct1(64 * 1024, 80, cgh);

    auto tex_acc = static_cast<dpct::image_wrapper<float, 1> *>(tex)->get_access(cgh);

    auto tex_smpl = tex->get_sampler();

    cgh.parallel_for(
        sycl::nd_range<3>(grid * block, block), [=](sycl::nd_item<3> item_ct1) {
          kernel(dpct::image_accessor_ext<float, 1>(tex_smpl, tex_acc),
                 item_ct1, stream_ct1);
        });
  });
}

int main() {
  dpct::device_ext &dev_ct1 = dpct::get_current_device();
  sycl::queue &q_ct1 = dev_ct1.default_queue();
  // declare and allocate memory
  float *buffer, *h_buffer;
  size_t pitch;
  buffer = (float *)dpct::dpct_malloc(pitch, N * sizeof(float), M);
  printf("pitch = %lu\n", pitch);
  q_ct1.memset(buffer, 0, M * pitch).wait();
  h_buffer=(float *)malloc(N*M*sizeof(float));
  for (int i = 0; i < M; i++)
    for (int j = 0; j < N; j++) h_buffer[i*N+j] = i+1;
  dpct::dpct_memcpy(buffer, pitch, h_buffer, N * sizeof(float),
                    N * sizeof(float), M, dpct::host_to_device);
  // create texture object
  dpct::image_data resDesc;
  memset(&resDesc, 0, sizeof(resDesc));
  resDesc.set_data_type(dpct::image_data_type::linear);
  resDesc.set_data_ptr(buffer);
  resDesc.set_channel_data_type(dpct::image_channel_data_type::fp);
  resDesc.set_channel_size(1, 32); // bits per channel
  resDesc.set_x(M * pitch * sizeof(float));

  dpct::sampling_info texDesc;
  memset(&texDesc, 0, sizeof(texDesc));
  /*
  DPCT1004:2: Compatible DPC++ code could not be generated.
  */
  texDesc.readMode = cudaReadModeElementType;

  // create texture object: we only have to do this once!
  dpct::image_wrapper_base_p tex = 0;
  tex = dpct::create_image_wrapper(resDesc, texDesc);

  call_kernel(tex); // pass texture as argument

  // destroy texture object
  delete tex;

  sycl::free(buffer, q_ct1);
}
