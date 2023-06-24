#include <cuda.h>
#include <stdio.h>
int main()
{
	cuInit( 0 );

	CUdevice cuDevice;
	CUresult res = cuDeviceGet( &cuDevice, 0 );
	if( res != CUDA_SUCCESS )
	{
		printf( "cannot acquire device 0\n" );
		exit( 1 );
	}

	CUcontext cuContext;
	res = cuCtxCreate( &cuContext, 0, cuDevice );
	if( res != CUDA_SUCCESS )
	{
		printf( "cannot create context\n" );
		exit( 1 );
	}


}