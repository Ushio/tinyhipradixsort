#include "shader.hpp"

#include <stdint.h>
#include <memory>
#include "Orochi/OrochiUtils.h"

#define THRS_KERNEL_FROM_FILE 1
#include "tinyhipradixsort.hpp"

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

// #define RADIX_SORT_TYPE uint64_t
#define RADIX_SORT_TYPE uint32_t

#define RADIX_SORT_VALUE_TYPE uint32_t

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

		std::vector<std::string> extraArgs;
		if( isNvidia )
		{
			extraArgs.push_back( ARG_DEBINFO_NV );

			// ITS enabled
			extraArgs.push_back( "--gpu-architecture=compute_70" );
		}
		else
		{
			extraArgs.push_back( ARG_DEBINFO_AMD );
		}
		thrs::RadixSort::Config config;
		switch (sizeof(RADIX_SORT_TYPE))
		{
		case 4:
			config.keyType = thrs::KeyType::U32;
			break;
		case 8:
			config.keyType = thrs::KeyType::U64;
			break;
		default:
			THRS_ASSERT( 1 );
		}
		thrs::RadixSort radixsort( extraArgs, config );

		// Shader shader( ( baseDir + "\\kernel.cu" ).c_str(), "kernel.cu", { baseDir }, {}, CompileMode::RelwithDebInfo, isNvidia );
		// std::vector<RADIX_SORT_TYPE> inputs( 1024 );
		std::vector<RADIX_SORT_TYPE> inputs( 160 * 1000 * 1000 );
		// std::vector<RADIX_SORT_TYPE> inputs( 1024 * 1024 * 128 );
		//  std::vector<RADIX_SORT_TYPE> inputs( 1024 * 1024 * 128 + 11 );
		// std::vector<RADIX_SORT_TYPE> inputs( 1024llu * 1024 * 1024 * 2 + 100 );

		std::vector<RADIX_SORT_VALUE_TYPE> inputValues( inputs.size() );

		splitmix64 rng;

		uint32_t numberOfInputs = inputs.size();

		std::unique_ptr<thrs::Buffer> inputKeyBuffer( new thrs::Buffer( sizeof( RADIX_SORT_TYPE ) * inputs.size() ) );
		std::unique_ptr<thrs::Buffer> inputValueBuffer( new thrs::Buffer( sizeof( RADIX_SORT_VALUE_TYPE ) * inputValues.size() ) );

		// column major
		// +---- buckets ( 256 ) ----
		// |
		// blocks
		// Buffer counterPrefixSumBuffer( radixsort.getTemporaryBufferBytes( numberOfInputs ).getTemporaryBufferBytesForSortKeys() );
		thrs::Buffer counterPrefixSumBuffer( radixsort.getTemporaryBufferBytes( numberOfInputs ).getTemporaryBufferBytesForSortPairs() );
		for (;;)
		{
			for( int i = 0; i < inputs.size(); i++ )
			{
				inputs[i] = rng.next();
				inputValues[i] = i;
			}

			oroMemcpyHtoDAsync( (oroDeviceptr)inputKeyBuffer->data(), inputs.data(), sizeof( RADIX_SORT_TYPE ) * inputs.size(), stream );
			oroMemcpyHtoDAsync( (oroDeviceptr)inputValueBuffer->data(), inputValues.data(), sizeof( RADIX_SORT_VALUE_TYPE ) * inputValues.size(), stream );

			OroStopwatch oroStream( stream );
			oroStream.start();

			// radixsort.sortKeys( inputKeyBuffer->data(), numberOfInputs, counterPrefixSumBuffer.data(), 0, sizeof( RADIX_SORT_TYPE ) * 8, stream );
			radixsort.sortPairs( inputKeyBuffer->data(), inputValueBuffer->data(), numberOfInputs, counterPrefixSumBuffer.data(), 0, sizeof( RADIX_SORT_TYPE ) * 8, stream );

			oroStream.stop();
			float ms = oroStream.getMs();
			oroStreamSynchronize( stream );

			printf( "%f ms\n", ms );

			std::vector<RADIX_SORT_TYPE> outputKeys( inputs.size() );
			std::vector<RADIX_SORT_VALUE_TYPE> outputValues( inputs.size() );
			oroMemcpyDtoH( outputKeys.data(), (oroDeviceptr)inputKeyBuffer->data(), sizeof( RADIX_SORT_TYPE ) * numberOfInputs );
			oroMemcpyDtoH( outputValues.data(), (oroDeviceptr)inputValueBuffer->data(), sizeof( RADIX_SORT_VALUE_TYPE ) * numberOfInputs );

			for( int i = 0; i < outputKeys.size() - 1; i++ )
			{
				SH_ASSERT( outputKeys[i] <= outputKeys[i + 1] );
			}

#if 0
			std::sort( inputs.begin(), inputs.end() );
			for( int i = 0; i < outputs.size(); i++ )
			{
				SH_ASSERT( outputs[i] == inputs[i] );
			}
#else
			std::vector<std::pair<RADIX_SORT_TYPE, uint32_t>> pairs( inputs.size() );
			for( int i = 0; i < inputs.size(); i++ )
			{
				pairs[i].first = inputs[i];
				pairs[i].second = inputValues[i];
			}
			std::stable_sort(pairs.begin(), pairs.end(), [](std::pair<RADIX_SORT_TYPE, uint32_t> a, std::pair<RADIX_SORT_TYPE, uint32_t> b ) {
				return a.first < b.first;
			});
			for( int i = 0; i < outputKeys.size(); i++ )
			{
				SH_ASSERT( outputKeys[i] == pairs[i].first );
				SH_ASSERT( outputValues[i] == pairs[i].second );
			}
#endif
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