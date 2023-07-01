#include <stdint.h>
#include <memory>
#include <algorithm>
#include <type_traits>
#include "Orochi/OrochiUtils.h"

#define THRS_KERNEL_FROM_FILE 1
#include "tinyhipradixsort.hpp"

#include "fpKey.hpp"

#include <ppl.h>
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

// using RADIX_SORT_KEY_TYPE = uint64_t;
using RADIX_SORT_KEY_TYPE = uint32_t;
// using RADIX_SORT_KEY_TYPE = float;
using RADIX_SORT_VALUE_TYPE = uint32_t;

//#define KEY_PAIR 1

template <
	typename T,
	typename std::enable_if<std::is_floating_point<T>::value, std::nullptr_t>::type = nullptr>
void varidateItem( T v )
{
	THRS_ASSERT( isfinite( v ) );
}
void varidateItem( ... )
{
}

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
		config.configureWithKeyPair<RADIX_SORT_KEY_TYPE, RADIX_SORT_VALUE_TYPE>();
		thrs::RadixSort radixsort( extraArgs, config );
		// std::vector<RADIX_SORT_KEY_TYPE> inputs( 8192 * 128 + 1 );
		//  std::vector<RADIX_SORT_KEY_TYPE> inputs( 1 );
		std::vector<RADIX_SORT_KEY_TYPE> inputs( 160 * 1000 * 1000 );
		// std::vector<RADIX_SORT_KEY_TYPE> inputs( 1024 * 1024 * 128 );
		//  std::vector<RADIX_SORT_KEY_TYPE> inputs( 1024 * 1024 * 128 + 11 );
		// std::vector<RADIX_SORT_KEY_TYPE> inputs( 1024llu * 1024 * 1024 * 2 + 100 );

		std::vector<RADIX_SORT_VALUE_TYPE> inputValues( inputs.size() );

		splitmix64 rng;

		uint32_t numberOfInputs = inputs.size();

		std::unique_ptr<thrs::Buffer> inputKeyBuffer( new thrs::Buffer( sizeof( RADIX_SORT_KEY_TYPE ) * inputs.size() ) );
		std::unique_ptr<thrs::Buffer> inputValueBuffer( new thrs::Buffer( sizeof( RADIX_SORT_VALUE_TYPE ) * inputValues.size() ) );

		// column major
		// +---- buckets ( 256 ) ----
		// |
		// blocks
#if defined( KEY_PAIR )
		thrs::Buffer tmpBuffer( radixsort.getTemporaryBufferBytes( numberOfInputs ).getTemporaryBufferBytesForSortPairs() );
#else
		thrs::Buffer tmpBuffer( radixsort.getTemporaryBufferBytes( numberOfInputs ).getTemporaryBufferBytesForSortKeys() );
#endif
		for (;;)
		{
			for( int i = 0; i < inputs.size(); i++ )
			{
				if ( std::is_same<RADIX_SORT_KEY_TYPE, float>::value )
				{
					uint32_t b = rng.next() & 0xFF7FFFFF;
					inputs[i] = *(RADIX_SORT_KEY_TYPE*)&b;
					varidateItem( inputs[i] );
				}
				else if ( std::is_same<RADIX_SORT_KEY_TYPE, double>::value )
				{
					uint64_t b = rng.next() & 0xFFEFFFFFFFFFFFFFllu;
					inputs[i] = *(RADIX_SORT_KEY_TYPE*)&b;
					varidateItem( inputs[i] );
				}
				else
				{
					inputs[i] = rng.next();
				}
				inputValues[i] = i;
			}

			oroMemcpyHtoDAsync( (oroDeviceptr)inputKeyBuffer->data(), inputs.data(), sizeof( RADIX_SORT_KEY_TYPE ) * inputs.size(), stream );
			oroMemcpyHtoDAsync( (oroDeviceptr)inputValueBuffer->data(), inputValues.data(), sizeof( RADIX_SORT_VALUE_TYPE ) * inputValues.size(), stream );

			OroStopwatch oroStream( stream );
			oroStream.start();

#if defined( KEY_PAIR )
			radixsort.sortPairs( inputKeyBuffer->data(), inputValueBuffer->data(), numberOfInputs, tmpBuffer.data(), 0, sizeof( RADIX_SORT_KEY_TYPE ) * 8, stream );
#else
			radixsort.sortKeys( inputKeyBuffer->data(), numberOfInputs, tmpBuffer.data(), 0, sizeof( RADIX_SORT_KEY_TYPE ) * 8, stream );
#endif

			oroStream.stop();
			float ms = oroStream.getMs();
			oroStreamSynchronize( stream );

			printf( "%f ms\n", ms );

			std::vector<RADIX_SORT_KEY_TYPE> outputKeys( inputs.size() );
			std::vector<RADIX_SORT_VALUE_TYPE> outputValues( inputs.size() );
			oroMemcpyDtoH( outputKeys.data(), (oroDeviceptr)inputKeyBuffer->data(), sizeof( RADIX_SORT_KEY_TYPE ) * numberOfInputs );
			oroMemcpyDtoH( outputValues.data(), (oroDeviceptr)inputValueBuffer->data(), sizeof( RADIX_SORT_VALUE_TYPE ) * numberOfInputs );

			for( int i = 0; i < outputKeys.size() - 1; i++ )
			{
				THRS_ASSERT( outputKeys[i] <= outputKeys[i + 1] );
			}

#if defined( KEY_PAIR )
			std::vector<std::pair<RADIX_SORT_KEY_TYPE, uint32_t>> pairs( inputs.size() );
			for( int i = 0; i < inputs.size(); i++ )
			{
				pairs[i].first = inputs[i];
				pairs[i].second = inputValues[i];
			}
			std::stable_sort(pairs.begin(), pairs.end(), [](std::pair<RADIX_SORT_KEY_TYPE, uint32_t> a, std::pair<RADIX_SORT_KEY_TYPE, uint32_t> b ) {
				return a.first < b.first;
			});
			for( int i = 0; i < outputKeys.size(); i++ )
			{
				THRS_ASSERT( outputKeys[i] == pairs[i].first );
				THRS_ASSERT( outputValues[i] == pairs[i].second );
			}
#else
			concurrency::parallel_radixsort( inputs.begin(), inputs.end(), []( RADIX_SORT_KEY_TYPE x ) { return getKeyBits( x ); } );

			// std::sort( inputs.begin(), inputs.end() );
			for( int i = 0; i < outputKeys.size(); i++ )
			{
				THRS_ASSERT( outputKeys[i] == inputs[i] );
			}
#endif
		}
	}

	oroCtxDestroy( ctx );

	return 0;
}