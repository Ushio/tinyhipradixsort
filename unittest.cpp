#include "utest.h"
// https://github.com/sheredom/utest.h

#include <Orochi/Orochi.h>
//#define THRS_KERNEL_FROM_FILE 1
#include "tinyhipradixsort.hpp"
#include <functional>
#include <algorithm>

#define TEST_ITERATION 32
#define TEST_MAX_ARRAY_SIZE 100000

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

int deviceIdx = 2;
oroDevice device;
oroStream stream;
oroCtx ctx;
std::vector<std::string> extraArgs;

int main( int argc, const char* const argv[] )
{
	if( oroInitialize( (oroApi)( ORO_API_HIP | ORO_API_CUDA ), 0 ) )
	{
		printf( "failed to init..\n" );
		return 0;
	}
	oroError err;
	err = oroInit( 0 );
	err = oroDeviceGet( &device, deviceIdx );
	err = oroCtxCreate( &ctx, 0, device );
	oroCtxSetCurrent( ctx );

	oroStreamCreate( &stream );
	oroDeviceProp props;
	oroGetDeviceProperties( &props, device );

	bool isNvidia = oroGetCurAPI( 0 ) & ORO_API_CUDADRIVER;

	printf( "Device: %s\n", props.name );
	printf( "Cuda: %s\n", isNvidia ? "Yes" : "No" );

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

	return utest_main( argc, argv );
}

UTEST_STATE();

template <class T>
void randomizeValues( splitmix64* rng, std::vector<T>* data )
{
	for( int i = 0; i < data->size(); i++ )
	{
		(*data)[i] = static_cast<T>( rng->next() );
	}
}

template <class KeyType>
void testSortKeys( std::function<void( bool )> assertion )
{
	thrs::RadixSort::Config config;
	config.configureWithKey<KeyType>();
	thrs::RadixSort radixsort( extraArgs, config );

	splitmix64 rng;
	for( int i = 0; i < TEST_ITERATION; i++ )
	{
		int numberOfInputs = 1 + rng.next() % ( TEST_MAX_ARRAY_SIZE - 1 );
		std::vector<KeyType> inputKeys( numberOfInputs );
		randomizeValues( &rng, &inputKeys );

		thrs::Buffer inputKeyBuffer( sizeof( KeyType ) * numberOfInputs );
		oroMemcpyHtoDAsync( (oroDeviceptr)inputKeyBuffer.data(), inputKeys.data(), sizeof( KeyType ) * inputKeys.size(), stream );

		thrs::Buffer tmpBuffer( radixsort.getTemporaryBufferBytes( numberOfInputs ).getTemporaryBufferBytesForSortKeys() );

		radixsort.sortKeys( inputKeyBuffer.data(), numberOfInputs, tmpBuffer.data(), 0, sizeof( KeyType ) * 8, stream );

		oroStreamSynchronize( stream );

		std::vector<KeyType> outputKeys( inputKeys.size() );
		oroMemcpyDtoH( outputKeys.data(), (oroDeviceptr)inputKeyBuffer.data(), sizeof( KeyType ) * numberOfInputs );

		std::sort( inputKeys.begin(), inputKeys.end() );

		for( int i = 0; i < inputKeys.size(); i++ )
		{
			assertion( inputKeys[i] == outputKeys[i] );
		}
	}
}

UTEST( SortKeys, u32 )
{
	using KeyType = uint32_t;
	testSortKeys<KeyType>( [&]( bool e )
						   { ASSERT_TRUE( e ); } );
}

UTEST( SortKeys, u64 )
{
	using KeyType = uint64_t;
	testSortKeys<KeyType>( [&]( bool e )
						{ ASSERT_TRUE( e ); } );
}

// == Pairs ==

template <class KeyType, class ValueType>
void testSortPairs( std::function<void(bool)> assertion )
{
	thrs::RadixSort::Config config;
	config.configureWithKeyPair<KeyType, ValueType>();
	thrs::RadixSort radixsort( extraArgs, config );

	splitmix64 rng;
	for( int i = 0; i < TEST_ITERATION; i++ )
	{
		int numberOfInputs = 1 + rng.next() % ( TEST_MAX_ARRAY_SIZE - 1 );
		std::vector<KeyType> inputKeys( numberOfInputs );
		std::vector<ValueType> inputValues( numberOfInputs );
		randomizeValues( &rng, &inputKeys );

		for( int i = 0; i < inputValues.size(); i++ )
		{
			inputValues[i] = ValueType( i );
		}

		thrs::Buffer inputKeyBuffer( sizeof( KeyType ) * numberOfInputs );
		oroMemcpyHtoDAsync( (oroDeviceptr)inputKeyBuffer.data(), inputKeys.data(), sizeof( KeyType ) * inputKeys.size(), stream );

		thrs::Buffer inputValueBuffer( sizeof( ValueType ) * numberOfInputs );
		oroMemcpyHtoDAsync( (oroDeviceptr)inputValueBuffer.data(), inputValues.data(), sizeof( ValueType ) * inputValues.size(), stream );

		thrs::Buffer tmpBuffer( radixsort.getTemporaryBufferBytes( numberOfInputs ).getTemporaryBufferBytesForSortPairs() );

		radixsort.sortPairs( inputKeyBuffer.data(), inputValueBuffer.data(), numberOfInputs, tmpBuffer.data(), 0, sizeof( KeyType ) * 8, stream );

		oroStreamSynchronize( stream );

		std::vector<KeyType> outputKeys( inputKeys.size() );
		oroMemcpyDtoH( outputKeys.data(), (oroDeviceptr)inputKeyBuffer.data(), sizeof( KeyType ) * numberOfInputs );
		std::vector<ValueType> outputValues( inputValues.size() );
		oroMemcpyDtoH( outputValues.data(), (oroDeviceptr)inputValueBuffer.data(), sizeof( ValueType ) * numberOfInputs );

		std::vector<std::pair<KeyType, ValueType>> pairs( inputKeys.size() );
		for( int i = 0; i < inputKeys.size(); i++ )
		{
			pairs[i].first = inputKeys[i];
			pairs[i].second = inputValues[i];
		}
		std::stable_sort( pairs.begin(), pairs.end(), []( std::pair<KeyType, ValueType> a, std::pair<KeyType, ValueType> b )
						  { return a.first < b.first; } );
		for( int i = 0; i < outputKeys.size(); i++ )
		{
			assertion( outputKeys[i] == pairs[i].first );
			assertion( outputValues[i] == pairs[i].second );
		}
	}
}

UTEST( SortPairs, K32V32 )
{
	using KeyType = uint32_t;
	using ValueType = uint32_t;
	testSortPairs<KeyType, ValueType>( [&]( bool e )
									   { ASSERT_TRUE( e ); } );
}
UTEST( SortPairs, K64V32 )
{
	using KeyType = uint64_t;
	using ValueType = uint32_t;
	testSortPairs<KeyType, ValueType>( [&]( bool e )
									   { ASSERT_TRUE( e ); } );
}
UTEST( SortPairs, K32V64 )
{
	using KeyType = uint32_t;
	using ValueType = uint64_t;
	testSortPairs<KeyType, ValueType>( [&]( bool e )
									   { ASSERT_TRUE( e ); } );
}
UTEST( SortPairs, K64V64 )
{
	using KeyType = uint64_t;
	using ValueType = uint64_t;
	testSortPairs<KeyType, ValueType>( [&]( bool e )
									   { ASSERT_TRUE( e ); } );
}

UTEST( SortPairs, K64V128 )
{
	struct u128
	{
		uint64_t a;
		uint64_t b;
		u128():a(0), b(0){}
		u128( uint64_t x ) : a( x ), b( x ) {}
		bool operator==(const u128& rhs) const
		{
			return a == rhs.a && b == rhs.b;
		}
	};

	using KeyType = uint64_t;
	using ValueType = u128;
	testSortPairs<KeyType, ValueType>( [&]( bool e )
									   { ASSERT_TRUE( e ); } );
}