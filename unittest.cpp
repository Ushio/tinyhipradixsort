
#include <Orochi/Orochi.h>
//#define THRS_KERNEL_FROM_FILE 1
#include "tinyhipradixsort.hpp"
#include <functional>
#include <algorithm>

#include <Windows.h>
#include <Orochi/OrochiUtils.h>
#include <ParallelPrimitives/RadixSort.h>
#include <ParallelPrimitives/RadixSortConfigs.h>

#include <ppl.h>

#include "utest.h"
// https://github.com/sheredom/utest.h

#define TEST_ITERATION 128
#define TEST_MAX_ARRAY_SIZE 100000
int deviceIdx = 0;

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

template <class T>
void sequentialValues( splitmix64* rng, std::vector<T>* data )
{
	for( int i = 0; i < data->size(); i++ )
	{
		( *data )[i] = static_cast<T>( i );
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


UTEST( StartBits, u64 )
{
	using KeyType = uint64_t;
	using ValueType = uint32_t;

	// sortKeys
	{
		thrs::RadixSort::Config config;
		config.configureWithKey<KeyType>();
		thrs::RadixSort radixsort( extraArgs, config );

		splitmix64 rng;

		for( int i = 0; i < TEST_ITERATION; i++ )
		{
			int numberOfInputs = 1 + rng.next() % ( TEST_MAX_ARRAY_SIZE - 1 );
			int startBit = rng.next() % 64;

			std::vector<KeyType> inputKeys( numberOfInputs );
			randomizeValues( &rng, &inputKeys );

			thrs::Buffer inputKeyBuffer( sizeof( KeyType ) * numberOfInputs );
			oroMemcpyHtoDAsync( (oroDeviceptr)inputKeyBuffer.data(), inputKeys.data(), sizeof( KeyType ) * inputKeys.size(), stream );

			thrs::Buffer tmpBuffer( radixsort.getTemporaryBufferBytes( numberOfInputs ).getTemporaryBufferBytesForSortKeys() );

			radixsort.sortKeys( inputKeyBuffer.data(), numberOfInputs, tmpBuffer.data(), startBit, startBit + 8, stream );

			oroStreamSynchronize( stream );

			std::vector<KeyType> outputKeys( inputKeys.size() );
			oroMemcpyDtoH( outputKeys.data(), (oroDeviceptr)inputKeyBuffer.data(), sizeof( KeyType ) * numberOfInputs );

			std::stable_sort( inputKeys.begin(), inputKeys.end(), [startBit]( KeyType a, KeyType b )
			{ 
				uint32_t bitA = ( a >> startBit ) & 0xFF;
				uint32_t bitB = ( b >> startBit ) & 0xFF;
				return bitA < bitB;
			} );

			for( int i = 0; i < inputKeys.size(); i++ )
			{
				ASSERT_TRUE( inputKeys[i] == outputKeys[i] );
			}
		}
	}

	// sortPairs
	thrs::RadixSort::Config config;
	config.configureWithKeyPair<KeyType, ValueType>();
	thrs::RadixSort radixsort( extraArgs, config );

	splitmix64 rng;
	for( int i = 0; i < TEST_ITERATION; i++ )
	{
		int numberOfInputs = 1 + rng.next() % ( TEST_MAX_ARRAY_SIZE - 1 );
		int startBit = rng.next() % 64;

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

		radixsort.sortPairs( inputKeyBuffer.data(), inputValueBuffer.data(), numberOfInputs, tmpBuffer.data(), startBit, startBit + 8, stream );

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
		std::stable_sort( pairs.begin(), pairs.end(), [startBit]( std::pair<KeyType, ValueType> a, std::pair<KeyType, ValueType> b )
		{ 
			uint32_t bitA = ( a.first >> startBit ) & 0xFF;
			uint32_t bitB = ( b.first >> startBit ) & 0xFF;
			return bitA < bitB; 
		} );
		for( int i = 0; i < outputKeys.size(); i++ )
		{
			ASSERT_TRUE( outputKeys[i] == pairs[i].first );
			ASSERT_TRUE( outputValues[i] == pairs[i].second );
		}
	}
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


UTEST( OrochiRadixSort, bench )
{
	int numberOfRun = 4;
	int numberOfInputs = 160 * 1000 * 1000;
	using KeyType = uint32_t;

	thrs::Buffer inputKeyBuffer( sizeof( KeyType ) * numberOfInputs );
	thrs::Buffer outputKeyBuffer( sizeof( KeyType ) * numberOfInputs );
	{
		OrochiUtils oroutils;
		splitmix64 rng;

		Oro::RadixSort m_sort;
		const auto s = m_sort.configure( device, oroutils, "../libs/Orochi/ParallelPrimitives/RadixSortKernels.h", "../libs/Orochi/" );
		uint32_t* m_tempBuffer;
		OrochiUtils::malloc( m_tempBuffer, s );

		for( int i = 0; i < numberOfRun; i++ )
		{
			std::vector<KeyType> inputKeys( numberOfInputs );
			randomizeValues( &rng, &inputKeys );
			oroMemcpyHtoDAsync( (oroDeviceptr)inputKeyBuffer.data(), inputKeys.data(), sizeof( KeyType ) * inputKeys.size(), stream );

			Oro::RadixSort::KeyValueSoA srcGpu{};
			Oro::RadixSort::KeyValueSoA dstGpu{};

			srcGpu.key = (uint32_t*)inputKeyBuffer.data();
			dstGpu.key = (uint32_t*)outputKeyBuffer.data();

			OroStopwatch oroStream( stream );
			oroStream.start();

			m_sort.sort( srcGpu.key, dstGpu.key, numberOfInputs, 0, 32, m_tempBuffer );

			oroStream.stop();
			float ms = oroStream.getMs();
			printf( "m_sort.sort %f ms\n", ms );

#if 1
			std::vector<KeyType> outputKeys( inputKeys.size() );
			oroMemcpyDtoH( outputKeys.data(), (oroDeviceptr)outputKeyBuffer.data(), sizeof( KeyType ) * numberOfInputs );

			concurrency::parallel_radixsort( inputKeys.begin(), inputKeys.end() );

			for( int i = 0; i < inputKeys.size(); i++ )
			{
				ASSERT_TRUE( inputKeys[i] == outputKeys[i] );
			}
#endif
		}

		OrochiUtils::free( m_tempBuffer );
	}

	{
		thrs::RadixSort::Config config;
		config.configureWithKey<KeyType>();
		thrs::RadixSort radixsort( extraArgs, config );
		thrs::Buffer tmpBuffer( radixsort.getTemporaryBufferBytes( numberOfInputs ).getTemporaryBufferBytesForSortKeys() );

		splitmix64 rng;
		for( int i = 0; i < numberOfRun; i++ )
		{
			std::vector<KeyType> inputKeys( numberOfInputs );
			randomizeValues( &rng, &inputKeys );
			oroMemcpyHtoDAsync( (oroDeviceptr)inputKeyBuffer.data(), inputKeys.data(), sizeof( KeyType ) * inputKeys.size(), stream );

			OroStopwatch oroStream( stream );
			oroStream.start();

			radixsort.sortKeys( inputKeyBuffer.data(), numberOfInputs, tmpBuffer.data(), 0, sizeof( KeyType ) * 8, stream );

			oroStream.stop();
			float ms = oroStream.getMs();
			printf( "radixsort.sortKeys %f ms\n", ms );
#if 1
			std::vector<KeyType> outputKeys( inputKeys.size() );
			oroMemcpyDtoH( outputKeys.data(), (oroDeviceptr)inputKeyBuffer.data(), sizeof( KeyType ) * numberOfInputs );

			concurrency::parallel_radixsort( inputKeys.begin(), inputKeys.end() );

			for( int i = 0; i < inputKeys.size(); i++ )
			{
				ASSERT_TRUE( inputKeys[i] == outputKeys[i] );
			}
#endif
		}
	}
}

template <class KeyType, class ValueType>
void stableSortPairs( std::vector<KeyType>* keys, std::vector<KeyType>* values )
{
	int numberOfInputs = keys->size();
	std::vector<std::pair<KeyType, ValueType>> pairs( numberOfInputs );
	for( int i = 0; i < numberOfInputs; i++ )
	{
		pairs[i].first = ( *keys )[i];
		pairs[i].second = ( *values )[i];
	}

	std::stable_sort( pairs.begin(), pairs.end(), []( std::pair<KeyType, ValueType> a, std::pair<KeyType, ValueType> b )
					  { return a.first < b.first; } );

	for( int i = 0; i < numberOfInputs; i++ )
	{
		( *keys )[i] = pairs[i].first;
		( *values )[i] = pairs[i].second;
	}
}

UTEST( OrochiRadixSort, benchKeyPair )
{
	int numberOfRun = 4;
	int numberOfInputs = 160 * 1000 * 1000;
	using KeyType = uint32_t;
	using ValueType = uint32_t;

	thrs::Buffer inputKeyBuffer( sizeof( KeyType ) * numberOfInputs );
	thrs::Buffer inputValueBuffer( sizeof( ValueType ) * numberOfInputs );
	{
		thrs::Buffer outputKeyBuffer( sizeof( KeyType ) * numberOfInputs );
		thrs::Buffer outputValueBuffer( sizeof( ValueType ) * numberOfInputs );

		OrochiUtils oroutils;
		splitmix64 rng;

		Oro::RadixSort m_sort;
		const auto s = m_sort.configure( device, oroutils, "../libs/Orochi/ParallelPrimitives/RadixSortKernels.h", "../libs/Orochi/" );
		uint32_t* m_tempBuffer;
		OrochiUtils::malloc( m_tempBuffer, s );

		for( int i = 0; i < numberOfRun; i++ )
		{
			std::vector<KeyType> inputKeys( numberOfInputs );
			randomizeValues( &rng, &inputKeys );
			oroMemcpyHtoDAsync( (oroDeviceptr)inputKeyBuffer.data(), inputKeys.data(), sizeof( KeyType ) * inputKeys.size(), stream );

			std::vector<KeyType> inputValues( numberOfInputs );
			sequentialValues( &rng, &inputValues );
			oroMemcpyHtoDAsync( (oroDeviceptr)inputValueBuffer.data(), inputValues.data(), sizeof( ValueType ) * inputValues.size(), stream );

			Oro::RadixSort::KeyValueSoA srcGpu{};
			Oro::RadixSort::KeyValueSoA dstGpu{};

			srcGpu.key = (uint32_t*)inputKeyBuffer.data();
			srcGpu.value = (uint32_t*)inputValueBuffer.data();
			dstGpu.key = (uint32_t*)outputKeyBuffer.data();
			dstGpu.value = (uint32_t*)outputValueBuffer.data();

			OroStopwatch oroStream( stream );
			oroStream.start();

			m_sort.sort( srcGpu, dstGpu, numberOfInputs, 0, 32, m_tempBuffer );

			oroStream.stop();
			float ms = oroStream.getMs();
			printf( "m_sort.sort %f ms\n", ms );

#if 1
			stableSortPairs<KeyType, ValueType>( &inputKeys, &inputValues );

			std::vector<KeyType> outputKeys( inputKeys.size() );
			std::vector<ValueType> outputValues( inputValues.size() );
			oroMemcpyDtoH( outputKeys.data(), (oroDeviceptr)outputKeyBuffer.data(), sizeof( KeyType ) * numberOfInputs );
			oroMemcpyDtoH( outputValues.data(), (oroDeviceptr)outputValueBuffer.data(), sizeof( ValueType ) * numberOfInputs );

			for( int i = 0; i < numberOfInputs; i++ )
			{
				ASSERT_TRUE( outputKeys[i] == inputKeys[i] );
				ASSERT_TRUE( outputValues[i] == inputValues[i] );
			}
#endif
		}

		OrochiUtils::free( m_tempBuffer );
	}

	{
		splitmix64 rng;

		thrs::RadixSort::Config config;
		config.configureWithKeyPair<KeyType, ValueType>();
		thrs::RadixSort radixsort( extraArgs, config );

		thrs::Buffer tmpBuffer( radixsort.getTemporaryBufferBytes( numberOfInputs ).getTemporaryBufferBytesForSortPairs() );

		for( int i = 0; i < numberOfRun; i++ )
		{
			std::vector<KeyType> inputKeys( numberOfInputs );
			randomizeValues( &rng, &inputKeys );
			oroMemcpyHtoDAsync( (oroDeviceptr)inputKeyBuffer.data(), inputKeys.data(), sizeof( KeyType ) * inputKeys.size(), stream );

			std::vector<KeyType> inputValues( numberOfInputs );
			sequentialValues( &rng, &inputValues );
			oroMemcpyHtoDAsync( (oroDeviceptr)inputValueBuffer.data(), inputValues.data(), sizeof( ValueType ) * inputValues.size(), stream );

			OroStopwatch oroStream( stream );
			oroStream.start();

			radixsort.sortPairs( inputKeyBuffer.data(), inputValueBuffer.data(), numberOfInputs, tmpBuffer.data(), 0, sizeof( KeyType ) * 8, stream );

			oroStream.stop();
			float ms = oroStream.getMs();
			printf( "radixsort.sortPairs %f ms\n", ms );

#if 1
			stableSortPairs<KeyType, ValueType>( &inputKeys, &inputValues );

			std::vector<KeyType> outputKeys( inputKeys.size() );
			std::vector<ValueType> outputValues( inputValues.size() );
			oroMemcpyDtoH( outputKeys.data(), (oroDeviceptr)inputKeyBuffer.data(), sizeof( KeyType ) * numberOfInputs );
			oroMemcpyDtoH( outputValues.data(), (oroDeviceptr)inputValueBuffer.data(), sizeof( ValueType ) * numberOfInputs );

			for( int i = 0; i < numberOfInputs; i++ )
			{
				ASSERT_TRUE( outputKeys[i] == inputKeys[i] );
				ASSERT_TRUE( outputValues[i] == inputValues[i] );
			}
#endif
		}
	}
}


UTEST( SortKeys, u32Large )
{
	using KeyType = uint32_t;
	thrs::RadixSort::Config config;
	config.configureWithKey<KeyType>();
	thrs::RadixSort radixsort( extraArgs, config );

	splitmix64 rng;
	uint32_t numberOfInputs = 1024llu * 1024 * 1024 * 2 + 100;
	thrs::Buffer tmpBuffer( radixsort.getTemporaryBufferBytes( numberOfInputs ).getTemporaryBufferBytesForSortKeys() );
	std::vector<KeyType> inputKeys( numberOfInputs );
	randomizeValues( &rng, &inputKeys );

	thrs::Buffer inputKeyBuffer( sizeof( KeyType ) * numberOfInputs );
	oroMemcpyHtoDAsync( (oroDeviceptr)inputKeyBuffer.data(), inputKeys.data(), sizeof( KeyType ) * inputKeys.size(), stream );

	radixsort.sortKeys( inputKeyBuffer.data(), numberOfInputs, tmpBuffer.data(), 0, sizeof( KeyType ) * 8, stream );

	oroStreamSynchronize( stream );

	std::vector<KeyType> outputKeys( inputKeys.size() );
	oroMemcpyDtoH( outputKeys.data(), (oroDeviceptr)inputKeyBuffer.data(), sizeof( KeyType ) * numberOfInputs );

	concurrency::parallel_sort( inputKeys.begin(), inputKeys.end() );

	for( int i = 0; i < inputKeys.size(); i++ )
	{
		ASSERT_TRUE( inputKeys[i] == outputKeys[i] );
	}
}