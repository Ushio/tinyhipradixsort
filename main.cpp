#include "shader.hpp"

int main()
{
	if( oroInitialize( (oroApi)( ORO_API_HIP | ORO_API_CUDA ), 0 ) )
	{
		printf( "failed to init..\n" );
		return 0;
	}
	int deviceIdx = 2;

	oroError err;
	err = oroInit( 0 );
	oroDevice device;
	err = oroDeviceGet( &device, deviceIdx );
	oroCtx ctx;
	err = oroCtxCreate( &ctx, 0, device );
	oroCtxSetCurrent( ctx );

	oroStream stream = 0;
	oroStreamCreate( &stream );
	oroDeviceProp props;
	oroGetDeviceProperties( &props, device );

	bool isNvidia = oroGetCurAPI( 0 ) & ORO_API_CUDADRIVER;

	printf( "Device: %s\n", props.name );
	printf( "Cuda: %s\n", isNvidia ? "Yes" : "No" );

	{
		std::string baseDir = "../"; /* repository root */
		Shader shader( ( baseDir + "\\kernel.cu" ).c_str(), "kernel.cu", { baseDir }, {}, CompileMode::RelwithDebInfo, isNvidia );

		ShaderArgument args;
		shader.launch( "kernelMain", args, 1, 1, 1, 1, 32, 1, stream );

		oroStreamSynchronize( stream );
	}

	oroCtxDestroy( ctx );

	return 0;
}