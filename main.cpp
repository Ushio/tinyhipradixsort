#include "shader.hpp"

#include <stdint.h>
#include <memory>
#include "Orochi/OrochiUtils.h"

class Stopwatch
{
public:
	using clock = std::chrono::high_resolution_clock;
	Stopwatch() : _started( clock::now() ) {}

	// seconds
	double elapsed() const
	{
		auto microseconds = std::chrono::duration_cast<std::chrono::microseconds>( clock::now() - _started ).count();
		return (double)microseconds * 0.001 * 0.001;
	}

private:
	clock::time_point _started;
};
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
inline uint32_t div_round_up( uint32_t val, uint32_t divisor )
{
	return ( val + divisor - 1 ) / divisor;
}
#define RADIX_SORT_BLOCK_SIZE 2048
#define RADIX_SORT_PREFIX_SCAN_BLOCK 8192
//#define RADIX_SORT_TYPE uint64_t
#define RADIX_SORT_TYPE uint32_t

static_assert( 32 <= RADIX_SORT_BLOCK_SIZE, "" );
static_assert( ( RADIX_SORT_BLOCK_SIZE % 32 ) == 0, "" );

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
		// std::vector<RADIX_SORT_TYPE> inputs( 1024 );
		std::vector<RADIX_SORT_TYPE> inputs( 160 * 1000 * 1000 );
		// std::vector<RADIX_SORT_TYPE> inputs( 1024 * 1024 * 128 + 11 );
		// std::vector<RADIX_SORT_TYPE> inputs( 1024llu * 1024 * 1024 * 2 + 100 );
		splitmix64 rng;

		uint32_t numberOfInputs = inputs.size();
		uint32_t numberOfBlocks = div_round_up( numberOfInputs, RADIX_SORT_BLOCK_SIZE );

		std::unique_ptr<Buffer> inputsBuffer( new Buffer( sizeof( RADIX_SORT_TYPE ) * inputs.size() ) );
		std::unique_ptr<Buffer> outputsBuffer( new Buffer( sizeof( RADIX_SORT_TYPE ) * inputs.size() ) );

		// column major
		// +---- buckets ( 256 ) ----
		// |
		// blocks
		Buffer counterPrefixSumBuffer( sizeof( uint32_t ) * 256 * numberOfBlocks );
		Buffer prefixSumIteratorBuffer( sizeof( uint32_t ) * 2 );

		for (;;)
		{
			for( int i = 0; i < inputs.size(); i++ )
			{
				inputs[i] = rng.next() & 0xFFFFFFFF;
			}

			oroMemcpyHtoDAsync( (oroDeviceptr)inputsBuffer->data(), inputs.data(), sizeof( RADIX_SORT_TYPE ) * inputs.size(), stream );

			OroStopwatch oroStream( stream );
			oroStream.start();
			for (int i = 0; i < 4; i++)
			{
				uint32_t bitLocation = i * 8;

				oroMemsetD32Async( (oroDeviceptr)prefixSumIteratorBuffer.data(), 0, prefixSumIteratorBuffer.bytes() / 4, stream );

				// counter
				{
					ShaderArgument args;
					args.add( inputsBuffer->data() );
					args.add( numberOfInputs );
					args.add( counterPrefixSumBuffer.data() );
					args.add( bitLocation );
					shader.launch( "blockCount", args, numberOfBlocks, 1, 1, 32, 1, 1, stream );
				}
				// Prefix Sum 
				{
					//OroStopwatch oroStream( stream );
					//oroStream.start();

					ShaderArgument args;
					args.add( counterPrefixSumBuffer.data() );
					args.add( numberOfBlocks * 256 );
					args.add( prefixSumIteratorBuffer.data() );
					// shader.launch( "prefixSumExclusive", args, 1, 1, 1, 32, 1, 1, stream );
					// printf( " prefixSumExclusive %d\n", numberOfBlocks * 256 / RADIX_SORT_PREFIX_SCAN_BLOCK );
					shader.launch( "prefixSumExclusiveInplace", args, div_round_up( numberOfBlocks * 256, RADIX_SORT_PREFIX_SCAN_BLOCK ), 1, 1, 32, 1, 1, stream );
					
					//oroStream.stop();
					//float ms = oroStream.getMs();
					//oroStreamSynchronize( stream );

					//printf( "pSum %f ms\n", ms );
				}
				// reorder
				{
					//OroStopwatch oroStream( stream );
					//oroStream.start();

					ShaderArgument args;
					args.add( inputsBuffer->data() );
					args.add( outputsBuffer->data() );
					args.add( numberOfInputs );
					args.add( counterPrefixSumBuffer.data() );
					// args.add( counterPrefixSumBuffer.data() );
					args.add( bitLocation );

					shader.launch( "reorder", args, numberOfBlocks, 1, 1, 32, 1, 1, stream );
				
					//oroStream.stop();
					//float ms = oroStream.getMs();
					//oroStreamSynchronize( stream );

					//printf( "reorder %f ms\n", ms );
				}

				std::swap( inputsBuffer, outputsBuffer );
			}

			oroStream.stop();
			float ms = oroStream.getMs();
			oroStreamSynchronize( stream );

			printf( "%f ms\n", ms );

			std::vector<RADIX_SORT_TYPE> outputs( inputs.size() );
			oroMemcpyDtoH( outputs.data(), (oroDeviceptr)inputsBuffer->data(), sizeof( RADIX_SORT_TYPE ) * numberOfInputs );

			for (int i = 0; i < outputs.size() - 1; i++)
			{
				SH_ASSERT( outputs[i] <= outputs[i + 1] );
			}
			Stopwatch sw;
			std::sort( inputs.begin(), inputs.end() );
			printf( "cpu %f ms,  %lld\n", sw.elapsed() * 1000.0, inputs[0] );
			for( int i = 0; i < outputs.size(); i++ )
			{
				SH_ASSERT( outputs[i] == inputs[i] );
			}
		}
		
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

		//std::vector<uint32_t> counterBufferFromGPU( 256 * numberOfBlocks );
		//oroMemcpyDtoH( counterBufferFromGPU.data(), (oroDeviceptr)counterBuffer.data(), sizeof( uint32_t ) * 256 * numberOfBlocks );

		////for (int i = 0; i < counterBufferRef.size(); i++)
		////{
		////	SH_ASSERT( counterBufferRef[i] == counterBufferFromGPU[i] );
		////}


		//std::vector<uint32_t> counterPrefixSumBufferFromGPU( 256 * numberOfBlocks );
		//oroMemcpyDtoH( counterPrefixSumBufferFromGPU.data(), (oroDeviceptr)counterPrefixSumBuffer.data(), sizeof( uint32_t ) * 256 * numberOfBlocks );

		//printf( "counterPrefixSumBufferFromGPU" );
	}

	oroCtxDestroy( ctx );

	return 0;
}