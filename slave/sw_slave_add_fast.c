#include <slave.h>
#include <simd.h>
#include <dma.h>
// BUFFSIZE: number of float/double numbers in LDM buffer
#define BUFFSIZE 4*1024
#define SIMDSIZE 4
#define SIMDTYPEF floatv4
#define SPNUM 64

__thread_local_fix dma_desc dma_get_src, dma_put_dst;

typedef struct addPara_st {
  void *src1;
  void *src2;
  void *dst;
  int count;
} addPara;

void sw_slave_add_fast(addPara *para) {
  float * local_src1 = (float *)ldm_malloc(BUFFSIZE*sizeof(float ));
  float * local_dst  = (float *)ldm_malloc(BUFFSIZE*sizeof(float ));
  SIMDTYPEF vsrc1,vsrc2;
  SIMDTYPEF vdst;
  int id = athread_get_id(-1);
  int count = para->count;
  int local_count = count/SPNUM + (id<(count%SPNUM));
  int start = id*(count/SPNUM) + (id<(count%SPNUM)?id:(count%SPNUM));
  float * src_ptr1 = &(((float *)para->src1)[start]);
  float * dst_ptr  = &(((float *)para->dst)[start]);
  volatile int replyget=0, replyput=0;
  int i,off;
  // DMA settings
  //dma_desc dma_get_src, dma_put_dst;
  int buff_size = local_count > 7680 ? 7680 : local_count;

  dma_set_op(&dma_get_src, DMA_GET);
  dma_set_mode(&dma_get_src, PE_MODE);
  dma_set_reply(&dma_get_src, &replyget);

  dma_set_op(&dma_put_dst, DMA_PUT);
  dma_set_mode(&dma_put_dst, PE_MODE);
  dma_set_reply(&dma_put_dst, &replyput);

  dma_set_size(&dma_get_src, buff_size*sizeof(float));
  dma_set_size(&dma_put_dst, buff_size*sizeof(float));

  for(off = 0; off+buff_size < local_count; off+=buff_size)
  {
    // DMA get a block
    dma(dma_get_src, (long)(src_ptr1+off), (long)(local_src1));
    dma(dma_get_src, (long)(dst_ptr+off), (long)(local_src2));
    dma_wait(&replyget, 2); replyget = 0;

    for(i=0; i<buff_size; i+=SIMDSIZE) {
      simd_load(vsrc1,&local_src1[i]);
      simd_load(vsrc2,&local_dst[i]);
      vdst = vsrc1 + vsrc2; // 
      simd_store(vdst,&local_dst[i]);
    }

    // DMA put result
    dma(dma_put_dst, (long)(dst_ptr+off), (long)(local_dst));
    dma_wait(&replyput, 1); replyput = 0;
  }

  if(off<local_count) {
    dma_set_size(&dma_get_src,(local_count-off)*sizeof(float));
    dma(dma_get_src, (long)(src_ptr1+off), (long)(local_src1));
    dma(dma_get_src, (long)(dst_ptr+off), (long)(local_src2));
    dma_wait(&replyget, 2); replyget = 0;

    for(i=0; i+SIMDSIZE-1<local_count-off; i+=SIMDSIZE) {
      simd_load(vsrc1,&local_src1[i]);
      simd_load(vsrc2,&local_src2[i]);
      vdst = vsrc1 + vsrc2; // 
      simd_store(vdst,&local_dst[i]);
    }
    for(;i<local_count-off;i++) {
      local_dst[i]=local_src1[i]+local_src2[i];
    }
    dma_set_size(&dma_put_dst,(local_count-off)*sizeof(float));
    dma(dma_put_dst, (long)(dst_ptr+off), (long)(local_dst));
    dma_wait(&replyput, 1); replyput = 0;
  }

  ldm_free(local_src1, BUFFSIZE*sizeof(float));
  ldm_free(local_src2, BUFFSIZE*sizeof(float));
  ldm_free(local_dst, BUFFSIZE*sizeof(float));
}

