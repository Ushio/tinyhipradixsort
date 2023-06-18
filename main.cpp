#include "shader.hpp"

#include <stdint.h>

struct splitmix64
{
	uint64_t x = 0; /* The state can be seeded with any value. */

	uint64_t next()
	{
		uint64_t z = ( x += 0x9e3779b97f4a7c15 );
		z = ( z ^ ( z >> 30 ) ) * 0xbf58476d1ce4e5b9;
		z = ( z ^ ( z >> 27 ) ) * 0x94d049bb133111eb;
		return z ^ ( z >> 31 );
	}
};
inline int div_round_up( int val, int divisor )
{
	return ( val + divisor - 1 ) / divisor;
}

#define RADIX_SORT_BLOCK_SIZE 1024
#define RADIX_SORT_PREFIX_SCAN_BLOCK 256

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

		std::vector<uint64_t> inputs( 1024 * 2 );

		splitmix64 rng;
		for( int i = 0; i < inputs.size(); i++ )
		{
			inputs[i] = rng.next();
		}

		uint64_t numberOfInputs = inputs.size();
		uint64_t numberOfBlocks = div_round_up( numberOfInputs, RADIX_SORT_BLOCK_SIZE );

		Buffer inputsBuffer( sizeof( uint64_t ) * inputs.size() );

		// column major
		// +---- buckets ( 256 ) ----
		// |
		// blocks
		Buffer counterBuffer( sizeof( uint32_t ) * 256 * numberOfBlocks );
		Buffer counterPrefixSumBuffer( sizeof( uint32_t ) * 256 * numberOfBlocks );
		Buffer prefixSumIteratorBuffer( sizeof( uint32_t ) );
		Buffer globalPrefixBuffer( sizeof( uint32_t ) );

		oroMemcpyHtoDAsync( (oroDeviceptr)inputsBuffer.data(), inputs.data(), sizeof( uint64_t ) * inputs.size(), stream );
		oroMemsetD32Async( (oroDeviceptr)prefixSumIteratorBuffer.data(), 0, 1, stream );
		oroMemsetD32Async( (oroDeviceptr)globalPrefixBuffer.data(), 0, 1, stream );

		oroStreamSynchronize( stream );

		{
			ShaderArgument args;
			args.add( inputsBuffer.data() );
			args.add( numberOfInputs );
			args.add( counterBuffer.data() );
			args.add( 0 );
			shader.launch( "blockCount", args, numberOfBlocks, 1, 1, RADIX_SORT_BLOCK_SIZE, 1, 1, stream );
		}

		// Prefix Sum 
		{
			ShaderArgument args;
			args.add( counterBuffer.data() );
			args.add( numberOfBlocks * 256 );
			args.add( counterPrefixSumBuffer.data() );
			args.add( prefixSumIteratorBuffer.data() );
			args.add( globalPrefixBuffer.data() );
			// shader.launch( "prefixSumExclusive", args, 1, 1, 1, 32, 1, 1, stream );
			shader.launch( "prefixSumExclusive", args, numberOfBlocks * 256 / RADIX_SORT_PREFIX_SCAN_BLOCK, 1, 1, 32, 1, 1, stream );
		}

		oroStreamSynchronize( stream );



		//std::vector<uint32_t> counterBufferRef( 256 * numberOfBlocks );
		//for( int i = 0; i < inputs.size(); i += RADIX_SORT_BLOCK_SIZE )
		//{
		//	for( int j = 0; j < RADIX_SORT_BLOCK_SIZE; j++ )
		//	{
		//		int bitLocation = 0;
		//		uint64_t item = inputs[i + j];
		//		uint32_t bits = ( item >> bitLocation ) & 0xFF;

		//		int bucketIndex = bits;
		//		int blockIndex = i / RADIX_SORT_BLOCK_SIZE;
		//		uint64_t counterIndex = bucketIndex * numberOfBlocks + blockIndex;
		//		counterBufferRef[counterIndex]++;
		//	}
		//}

		std::vector<uint32_t> counterBufferFromGPU( 256 * numberOfBlocks );
		oroMemcpyDtoH( counterBufferFromGPU.data(), (oroDeviceptr)counterBuffer.data(), sizeof( uint32_t ) * 256 * numberOfBlocks );

		//for (int i = 0; i < counterBufferRef.size(); i++)
		//{
		//	SH_ASSERT( counterBufferRef[i] == counterBufferFromGPU[i] );
		//}


		std::vector<uint32_t> counterPrefixSumBufferFromGPU( 256 * numberOfBlocks );
		oroMemcpyDtoH( counterPrefixSumBufferFromGPU.data(), (oroDeviceptr)counterPrefixSumBuffer.data(), sizeof( uint32_t ) * 256 * numberOfBlocks );

		printf( "counterPrefixSumBufferFromGPU" );
	}

	oroCtxDestroy( ctx );

	return 0;
}