#include "Orochi/Orochi.h"
#include <memory>
#include <stdint.h>
#include <vector>
#include <random>

#include "tinyhipradixsort.hpp"

int main()
{
	if( oroInitialize( (oroApi)( ORO_API_HIP | ORO_API_CUDA ), 0 ) )
	{
		printf( "failed to init..\n" );
		return 0;
	}
	int deviceIdx = 0;

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
		// Initialize shader
		std::vector<std::string> extraArgs;
		thrs::RadixSort::Config config;
		config.configureWithKey<uint32_t>();
		thrs::RadixSort radixsort( extraArgs, config );

		// Prepare Inputs
		std::vector<uint32_t> inputs( 32 );
		for( int i = 0; i < inputs.size(); i++ )
		{
			inputs[i] = rand();
		}

		uint32_t numberOfInputs = inputs.size();
		std::unique_ptr<thrs::Buffer> inputKeyBuffer( new thrs::Buffer( sizeof( uint32_t ) * inputs.size() ) );
		oroMemcpyHtoDAsync( (oroDeviceptr)inputKeyBuffer->data(), inputs.data(), sizeof( uint32_t ) * inputs.size(), stream );

		// Temporal buffer allocation
		thrs::Buffer tmpBuffer( radixsort.getTemporaryBufferBytes( numberOfInputs ).getTemporaryBufferBytesForSortKeys() );
		
		// Sort
		radixsort.sortKeys( inputKeyBuffer->data(), numberOfInputs, tmpBuffer.data(), 0, 32, stream );

		// Readback
		std::vector<uint32_t> outputKeys( inputs.size() );
		oroMemcpyDtoH( outputKeys.data(), (oroDeviceptr)inputKeyBuffer->data(), sizeof( uint32_t ) * numberOfInputs );

		for (auto v : outputKeys)
		{
			printf( "%u\n", v );
		}

		oroCtxDestroy( ctx );
	}

	return 0;
}