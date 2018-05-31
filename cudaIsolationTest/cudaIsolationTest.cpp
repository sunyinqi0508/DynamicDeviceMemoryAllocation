// cudaIsolationTest.cpp : Defines the entry point for the console application.
//

//#include "stdafx.h"
#include "cuTest.cuh"
#include <malloc.h>
#include <stdio.h>
int main()
{
	Bridge *b = new Bridge();// (Bridge*)malloc(sizeof(Bridge));
	msvcInit_vc();
	init();
	cutest_launch (128, 128, b);
	cudasync();
	printf("results %d %f\n", b->a, b->b);
    return 0;
}

 