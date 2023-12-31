typedef unsigned long long uint64_t;
typedef unsigned int uint32_t;
typedef unsigned short uint16_t;
typedef unsigned char uint8_t;

#define RADIX_SORT_BLOCK_SIZE 2048
#define RADIX_SORT_PREFIX_SCAN_BLOCK 16384

#define BLOCK_COUNT_NUMBER_OF_WARPS 8
#define BLOCK_COUNT_NUMBER_OF_THREADS_PER_BLOCK ( 32 * BLOCK_COUNT_NUMBER_OF_WARPS )

#define PSUM_NUMBER_OF_WARPS 16
#define PSUM_NUMBER_OF_THREADS_PER_BLOCK ( 32 * PSUM_NUMBER_OF_WARPS )

#define REORDER_NUMBER_OF_WARPS 8
#define REORDER_NUMBER_OF_THREADS_PER_BLOCK ( 32 * REORDER_NUMBER_OF_WARPS )

#if defined( DESCENDING_ORDER )
#define ORDER_MASK_32 0xFFFFFFFF
#define ORDER_MASK_64 0xFFFFFFFFFFFFFFFFllu
#else
#define ORDER_MASK_32 0
#define ORDER_MASK_64 0llu
#endif

#if defined( CUDART_VERSION ) && CUDART_VERSION >= 9000
	#define ITS 1
#endif

__device__ inline uint32_t div_round_up( uint32_t val, uint32_t divisor )
{
	return ( val + divisor - 1 ) / divisor;
}
template <int NElement, int NThread, class T>
__device__ void clearShared( T* sMem, T value )
{
	for( int i = 0; i < NElement; i += NThread )
	{
		if( i < NElement )
		{
			sMem[i + threadIdx.x] = value;
		}
	}
}

__device__ inline uint32_t getKeyBits( uint32_t x )
{
	return x ^ ORDER_MASK_32;
}
__device__ inline uint64_t getKeyBits( uint64_t x )
{
	return x ^ ORDER_MASK_64;
}
__device__ inline uint32_t getKeyBits( float x )
{
	if( x == 0.0f )
		x = 0.0f;

	uint32_t flip = uint32_t( __float_as_int( x ) >> 31 ) | 0x80000000;
	return __float_as_uint( x ) ^ flip ^ ORDER_MASK_32;
}
__device__ inline uint64_t getKeyBits( double x )
{
	if( x == 0.0 )
		x = 0.0;

	uint64_t flip = uint64_t( __double_as_longlong( x ) >> 63 ) | 0x8000000000000000llu;
	return (uint64_t)__double_as_longlong( x ) ^ flip ^ ORDER_MASK_64;
}

__device__ uint64_t g_iterator;

extern "C" __global__ void blockCount( RADIX_SORT_KEY_TYPE* inputs, uint32_t numberOfInputs, uint32_t* counters, uint32_t bitLocation )
{
	__shared__ uint32_t localCounters[256];
	clearShared<256, BLOCK_COUNT_NUMBER_OF_THREADS_PER_BLOCK, uint32_t>( localCounters, 0 );
	__syncthreads();

	for( int i = 0; i < RADIX_SORT_BLOCK_SIZE; i += BLOCK_COUNT_NUMBER_OF_THREADS_PER_BLOCK )
	{
		uint32_t itemIndex = blockIdx.x * RADIX_SORT_BLOCK_SIZE + threadIdx.x + i;
		if( itemIndex < numberOfInputs )
		{
			auto item = inputs[itemIndex];
			uint32_t bits = ( getKeyBits( item ) >> bitLocation ) & 0xFF;
			atomicInc( &localCounters[bits], 0xFFFFFFFF );
		}
	}

	__syncthreads();

	for( int i = 0; i < 256; i += BLOCK_COUNT_NUMBER_OF_THREADS_PER_BLOCK )
	{
		uint32_t numberOfBlocks = div_round_up( numberOfInputs, RADIX_SORT_BLOCK_SIZE );
		uint32_t bucketIndex = i + threadIdx.x;
		uint32_t blockIndex = blockIdx.x;
		uint32_t counterIndex = bucketIndex * numberOfBlocks + blockIndex;
		counters[counterIndex] = localCounters[bucketIndex];
	}

	if( blockIdx.x == 0 && threadIdx.x == 0 )
		g_iterator = 0;
}

template <int NThreads>
__device__ inline uint32_t prefixSumExclusive( uint32_t prefix, uint32_t* sMemIO )
{
	uint32_t value = sMemIO[threadIdx.x];

	for( uint32_t offset = 1; offset < NThreads; offset <<= 1 )
	{
		uint32_t x = sMemIO[threadIdx.x];

		if( offset <= threadIdx.x )
		{
			x += sMemIO[threadIdx.x - offset];
		}

		__syncthreads();

		sMemIO[threadIdx.x] = x;

		__syncthreads();
	}
	uint32_t sum = sMemIO[NThreads - 1];

	__syncthreads();

	sMemIO[threadIdx.x] += prefix - value;

	__syncthreads();

	return sum;
}

extern "C" __global__ void prefixSumExclusiveInplace( uint32_t* inout, uint32_t numberOfInputs )
{
	__shared__ uint32_t gp;
	__shared__ uint32_t smem[PSUM_NUMBER_OF_THREADS_PER_BLOCK];

	uint32_t blockIndex = blockIdx.x;

	__syncthreads();

	uint32_t s = 0;
	for( int i = 0; i < RADIX_SORT_PREFIX_SCAN_BLOCK; i += PSUM_NUMBER_OF_THREADS_PER_BLOCK )
	{
		uint32_t itemIndex = blockIndex * RADIX_SORT_PREFIX_SCAN_BLOCK + i + threadIdx.x;
		s += itemIndex < numberOfInputs ? inout[itemIndex] : 0;
	}

	smem[threadIdx.x] = s;

	__syncthreads();

	for( int i = 1; i < PSUM_NUMBER_OF_THREADS_PER_BLOCK; i <<= 1 )
	{
		uint32_t a = smem[threadIdx.x];
		uint32_t b = smem[threadIdx.x ^ i];
		__syncthreads();
		smem[threadIdx.x] = a + b;
		__syncthreads();
	}

	if( threadIdx.x == 0 )
	{
		uint32_t prefix = smem[0];

		uint64_t expected;
		uint64_t cur = g_iterator;
		uint32_t globalPrefix = cur & 0xFFFFFFFF;
		do
		{
			expected = (uint64_t)globalPrefix + ( (uint64_t)( blockIndex ) << 32 );
			uint64_t newValue = (uint64_t)globalPrefix + prefix | ( (uint64_t)( blockIndex + 1 ) << 32 );
			cur = atomicCAS( &g_iterator, expected, newValue );
			globalPrefix = cur & 0xFFFFFFFF;

		} while( cur != expected );

		gp = globalPrefix;
	}

	__syncthreads();

	uint32_t globalPrefix = gp;

	for( int i = 0; i < RADIX_SORT_PREFIX_SCAN_BLOCK; i += PSUM_NUMBER_OF_THREADS_PER_BLOCK )
	{
		uint32_t itemIndex = blockIndex * RADIX_SORT_PREFIX_SCAN_BLOCK + i + threadIdx.x;
		smem[threadIdx.x] = itemIndex < numberOfInputs ? inout[itemIndex] : 0;

		__syncthreads();

		globalPrefix += prefixSumExclusive<PSUM_NUMBER_OF_THREADS_PER_BLOCK>( globalPrefix, smem );

		if( itemIndex < numberOfInputs )
		{
			inout[itemIndex] = smem[threadIdx.x];
		}

		__syncthreads();
	}
}

__device__ __forceinline__ void reorder( RADIX_SORT_KEY_TYPE* inputKeys, RADIX_SORT_KEY_TYPE* outputKeys, RADIX_SORT_VALUE_TYPE* inputValues, RADIX_SORT_VALUE_TYPE* outputValues, bool keyPair, uint32_t numberOfInputs, uint32_t* sums, uint32_t bitLocation )
{
#if 1
	struct ElementLocation
	{
		uint32_t localSrcIndex : 12;
		uint32_t localOffset : 12;
		uint32_t bucket : 8;
	};

	__shared__ uint32_t pSum[256];
	__shared__ uint32_t localPrefixSum[256];
	__shared__ uint32_t counters[256];
	__shared__ ElementLocation elementLocations[RADIX_SORT_BLOCK_SIZE];
	__shared__ uint8_t elementBuckets[RADIX_SORT_BLOCK_SIZE];
	__shared__ uint32_t matchMasks[REORDER_NUMBER_OF_WARPS][256];

	uint32_t blockIndex = blockIdx.x;
	uint32_t numberOfBlocks = div_round_up( numberOfInputs, RADIX_SORT_BLOCK_SIZE );

	clearShared<256, REORDER_NUMBER_OF_THREADS_PER_BLOCK, uint32_t>( localPrefixSum, 0 );
	clearShared<256, REORDER_NUMBER_OF_THREADS_PER_BLOCK, uint32_t>( counters, 0 );

	for( int i = 0; i < 256; i += REORDER_NUMBER_OF_THREADS_PER_BLOCK )
	{
		uint32_t counterIndex = ( i + threadIdx.x ) /* bucketIndex */ * numberOfBlocks + blockIndex;
		pSum[i + threadIdx.x] = sums[counterIndex];
	}

	for( int w = 0; w < REORDER_NUMBER_OF_WARPS; w++ )
	{
		for( int i = 0; i < 256; i += REORDER_NUMBER_OF_THREADS_PER_BLOCK )
		{
			matchMasks[w][i + threadIdx.x] = 0;
		}
	}

	__syncthreads();

	// count
#if defined( KEY_IS_16BYTE_ALIGNED )
	if( ( blockIndex + 1 ) * RADIX_SORT_BLOCK_SIZE <= numberOfInputs )
	{
		for( int i = 0; i < RADIX_SORT_BLOCK_SIZE; i += REORDER_NUMBER_OF_THREADS_PER_BLOCK * 4 )
		{
			uint32_t itemIndex = blockIndex * RADIX_SORT_BLOCK_SIZE + i + threadIdx.x * 4;
			struct alignas( 16 ) Key4
			{
				RADIX_SORT_KEY_TYPE xs[4];
			};
			Key4 key4 = *(Key4 *)&inputKeys[itemIndex];
			for (int k = 0; k < 4; k++)
			{
				auto item = key4.xs[k];
				uint32_t bucketIndex = ( getKeyBits( item ) >> bitLocation ) & 0xFF;
				atomicInc( &localPrefixSum[bucketIndex], 0xFFFFFFFF );
				elementBuckets[i + threadIdx.x * 4 + k] = bucketIndex;
			}
		}
	}
	else
#endif
	{
		for( int i = 0; i < RADIX_SORT_BLOCK_SIZE; i += REORDER_NUMBER_OF_THREADS_PER_BLOCK )
		{
			uint32_t itemIndex = blockIndex * RADIX_SORT_BLOCK_SIZE + i + threadIdx.x;
			if( itemIndex < numberOfInputs )
			{
				auto item = inputKeys[itemIndex];
				uint32_t bucketIndex = ( getKeyBits( item ) >> bitLocation ) & 0xFF;
				atomicInc( &localPrefixSum[bucketIndex], 0xFFFFFFFF );

				elementBuckets[i + threadIdx.x] = bucketIndex;
			}
		}
	}

	// prefix sum
	__syncthreads();

	uint32_t prefix = 0;
	for( int i = 0; i < 256; i += REORDER_NUMBER_OF_THREADS_PER_BLOCK )
	{
		prefix += prefixSumExclusive<REORDER_NUMBER_OF_THREADS_PER_BLOCK>( prefix, &localPrefixSum[i] );
	}

	// reorder
	for( int i = 0; i < RADIX_SORT_BLOCK_SIZE; i += REORDER_NUMBER_OF_THREADS_PER_BLOCK )
	{
		uint32_t itemIndex = blockIndex * RADIX_SORT_BLOCK_SIZE + i + threadIdx.x;
		uint32_t bucketIndex = elementBuckets[i + threadIdx.x];

		__syncthreads();

		int warp = threadIdx.x / 32;
		int lane = threadIdx.x % 32;
		
		if( itemIndex < numberOfInputs )
		{
			atomicOr( &matchMasks[warp][bucketIndex], 1u << lane );
		}

		__syncthreads();

		bool flushMask = false;

		if( itemIndex < numberOfInputs )
		{
			uint32_t matchMask = matchMasks[warp][bucketIndex];
			uint32_t lowerMask = ( 1u << lane ) - 1;
			uint32_t offset = __popc( matchMask & lowerMask );

			flushMask = offset == 0;

			for( int w = 0; w < warp; w++ )
			{
				offset += __popc( matchMasks[w][bucketIndex] );
			}

			uint32_t localOffset = counters[bucketIndex] + offset;
			uint32_t to = localOffset + localPrefixSum[bucketIndex];

			ElementLocation el;
			el.localSrcIndex = i + threadIdx.x;
			el.localOffset = localOffset;
			el.bucket = bucketIndex;
			elementLocations[to] = el;
		}

		__syncthreads();

		if( itemIndex < numberOfInputs )
		{
			atomicInc( &counters[bucketIndex], 0xFFFFFFFF );
		}
		if( flushMask )
		{
			matchMasks[warp][bucketIndex] = 0;
		}
	}

	for( int i = 0; i < RADIX_SORT_BLOCK_SIZE; i += REORDER_NUMBER_OF_THREADS_PER_BLOCK )
	{
		uint32_t itemIndex = blockIndex * RADIX_SORT_BLOCK_SIZE + i + threadIdx.x;
		if( itemIndex < numberOfInputs )
		{
			ElementLocation el = elementLocations[i + threadIdx.x];
			uint32_t srcIndex = blockIndex * RADIX_SORT_BLOCK_SIZE + el.localSrcIndex;
			uint8_t bucketIndex = el.bucket;

			uint32_t dstIndex = pSum[bucketIndex] + el.localOffset;
			outputKeys[dstIndex] = inputKeys[srcIndex];
		}
	}

	if( keyPair )
	{
		for( int i = 0; i < RADIX_SORT_BLOCK_SIZE; i += REORDER_NUMBER_OF_THREADS_PER_BLOCK )
		{
			uint32_t itemIndex = blockIndex * RADIX_SORT_BLOCK_SIZE + i + threadIdx.x;
			if( itemIndex < numberOfInputs )
			{
				ElementLocation el = elementLocations[i + threadIdx.x];
				uint32_t srcIndex = blockIndex * RADIX_SORT_BLOCK_SIZE + el.localSrcIndex;
				uint8_t bucketIndex = el.bucket;

				uint32_t dstIndex = pSum[bucketIndex] + el.localOffset;
				outputValues[dstIndex] = inputValues[srcIndex];
			}
		}
	}
#else
	__shared__ uint32_t psum[256];

	uint32_t blockIndex = blockIdx.x;
	uint32_t numberOfBlocks = div_round_up( numberOfInputs, RADIX_SORT_BLOCK_SIZE );
	for( int i = 0; i < 256; i += REORDER_NUMBER_OF_THREADS_PER_BLOCK )
	{
		uint32_t counterIndex = ( i + threadIdx.x ) /* bucketIndex */ * numberOfBlocks + blockIndex;
		psum[i + threadIdx.x] = sums[counterIndex];
	}
	__syncthreads();

	for( int i = 0; i < RADIX_SORT_BLOCK_SIZE; i += REORDER_NUMBER_OF_THREADS_PER_BLOCK )
	{
		uint32_t itemIndex = blockIndex * RADIX_SORT_BLOCK_SIZE + i + threadIdx.x;
		RADIX_SORT_KEY_TYPE item;
		uint32_t bucketIndex;
		if( itemIndex < numberOfInputs )
		{
			item = inputKeys[itemIndex];
			bucketIndex = ( getKeyBits( item ) >> bitLocation ) & 0xFF;
		}
		uint32_t location = 0xFFFFFFFF;
		for( int j = 0; j < REORDER_NUMBER_OF_THREADS_PER_BLOCK; j++ )
		{
			if( j == threadIdx.x )
			{
				if( itemIndex < numberOfInputs )
				{
					location = psum[bucketIndex]++;
				}
			}
			__syncthreads();
		}
		if( location != 0xFFFFFFFF )
		{
			outputKeys[location] = item;
			if( keyPair )
			{
				outputValues[location] = inputValues[itemIndex];
			}
		}
	}
#endif
}
extern "C" __global__ void reorderKey( RADIX_SORT_KEY_TYPE* inputKeys, RADIX_SORT_KEY_TYPE* outputKeys, uint32_t numberOfInputs, uint32_t* sums, uint32_t bitLocation )
{
	reorder( inputKeys, outputKeys, nullptr, nullptr, false, numberOfInputs, sums, bitLocation );
}
extern "C" __global__ void reorderKeyPair( RADIX_SORT_KEY_TYPE* inputKeys, RADIX_SORT_KEY_TYPE* outputKeys, RADIX_SORT_VALUE_TYPE* inputValues, RADIX_SORT_VALUE_TYPE* outputValues, uint32_t numberOfInputs, uint32_t* sums, uint32_t bitLocation )
{
	reorder( inputKeys, outputKeys, inputValues, outputValues, true, numberOfInputs, sums, bitLocation );
}