#include <cstdio>
#ifdef __CUDA_ARCH__

#else
#define __device__
#endif
struct Bridge {
public:
	int a;
	float b;
	__device__
	Bridge() {
		a = 0;
		b = 0;
#ifdef __CUDACC__
		printf("InitBridge in CUDA\n");
#else
		printf("InitBridge in HOST\n");
#endif

	}
	

};
#ifdef __CUDACC__
void msvcInit() 
#else
void msvcInit_vc() 
#endif
{
	printf("MSVC_INIT: %ld %ld\n", __cplusplus, _MSVC_LANG);
}
void cutest_launch(int griddim, int blockdim, Bridge* b);
void cudasync();
void init();