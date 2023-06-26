typedef unsigned long long uint64_t;
typedef unsigned int uint32_t;
typedef unsigned short uint16_t;
typedef unsigned char uint8_t;

#define BLOCK_COUNT_NUMBER_OF_WARPS 4
#define BLOCK_COUNT_NUMBER_OF_THREADS_PER_BLOCK ( 32 * BLOCK_COUNT_NUMBER_OF_WARPS )

#define REORDER_NUMBER_OF_WARPS 8
#define REORDER_NUMBER_OF_THREADS_PER_BLOCK ( 32 * REORDER_NUMBER_OF_WARPS )


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
			uint32_t bits = ( item >> bitLocation ) & 0xFF;
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

__device__ void warpPrefixSumExclusive( uint32_t val, uint32_t* p, uint32_t* sum )
{
	uint32_t x = val;

	for( uint32_t offset = 1; offset < 32; offset <<= 1 )
	{
#if defined( ITS )
		__syncwarp( 0xffffffff );
		uint32_t y = __shfl_up_sync( 0xffffffff, x, offset );
#else
		uint32_t y = __shfl_up( x, offset );
#endif

		if( offset <= threadIdx.x )
		{
			x += y;
		}
	}
#if defined( ITS )
	__syncwarp( 0xffffffff );
	*sum = __shfl_sync( 0xffffffff, x, 31 );
#else
	__syncthreads();
	*sum = __shfl( x, 31 );
#endif
	*p = x - val;
}

extern "C" __global__ void prefixSumExclusiveInplace( uint32_t* inout, uint32_t numberOfInputs )
{
	__shared__ uint32_t localPrefixSum[RADIX_SORT_PREFIX_SCAN_BLOCK];

	uint32_t blockIndex = blockIdx.x;
	uint32_t prefix = 0;
	for( int i = 0; i < RADIX_SORT_PREFIX_SCAN_BLOCK; i += 32 )
	{
		uint32_t itemIndex = blockIndex * RADIX_SORT_PREFIX_SCAN_BLOCK + i + threadIdx.x;
		uint32_t p;
		uint32_t s;
		warpPrefixSumExclusive( itemIndex < numberOfInputs ? inout[itemIndex] : 0, &p, &s );
		localPrefixSum[i + threadIdx.x] = prefix + p;
		prefix += s;
	}

	int gp;
	if( threadIdx.x == 0 )
	{
		uint64_t expected;
		uint64_t cur = g_iterator;
		gp = cur & 0xFFFFFFFF;
		do
		{
			expected = (uint64_t)gp + ( (uint64_t)( blockIndex ) << 32 );
			uint64_t newValue = (uint64_t)gp + prefix | ( (uint64_t)( blockIndex + 1 ) << 32 );
			cur = atomicCAS( &g_iterator, expected, newValue );
			gp = cur & 0xFFFFFFFF;
		} while( cur != expected );
	}

#if defined( ITS )
	__syncwarp( 0xffffffff );
	gp = __shfl_sync( 0xffffffff, gp, 0 );
#else
	__syncthreads();
	gp = __shfl( gp, 0 );
#endif

	for( int i = 0; i < RADIX_SORT_PREFIX_SCAN_BLOCK; i += 32 )
	{
		uint32_t itemIndex = blockIndex * RADIX_SORT_PREFIX_SCAN_BLOCK + i + threadIdx.x;
		if (itemIndex < numberOfInputs)
		{
			inout[itemIndex] = gp + localPrefixSum[i + threadIdx.x];
		}
	}
}

template <int NThreads>
__device__ uint32_t prefixSumExclusive( uint32_t prefix, uint32_t* sMemIO )
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

__device__ __forceinline__ void reorder( RADIX_SORT_KEY_TYPE* inputKeys, RADIX_SORT_KEY_TYPE* outputKeys, RADIX_SORT_VALUE_TYPE* inputValues, RADIX_SORT_VALUE_TYPE* outputValues, bool keyPair, uint32_t numberOfInputs, uint32_t* sums, uint32_t bitLocation )
{
#if 1
	__shared__ uint32_t localPrefixSum[256];
	__shared__ uint16_t elementIndices[RADIX_SORT_BLOCK_SIZE];
	__shared__ uint32_t matchMasks[REORDER_NUMBER_OF_WARPS][256];

	uint32_t blockIndex = blockIdx.x;
	uint32_t numberOfBlocks = div_round_up( numberOfInputs, RADIX_SORT_BLOCK_SIZE );
	clearShared<256, REORDER_NUMBER_OF_THREADS_PER_BLOCK, uint32_t>( localPrefixSum, 0 );
	__syncthreads();

	// count
	for( int i = 0; i < RADIX_SORT_BLOCK_SIZE; i += REORDER_NUMBER_OF_THREADS_PER_BLOCK )
	{
		uint32_t itemIndex = blockIndex * RADIX_SORT_BLOCK_SIZE + i + threadIdx.x;
		if( itemIndex < numberOfInputs )
		{
			auto item = inputKeys[itemIndex];
			uint32_t bucketIndex = ( item >> bitLocation ) & 0xFF;
			atomicInc( &localPrefixSum[bucketIndex], 0xFFFFFFFF );
		}
	}

	// prefix sum
	__syncthreads();

	uint32_t prefix = 0;
	for( int i = 0; i < 256; i += REORDER_NUMBER_OF_THREADS_PER_BLOCK )
	{
		prefix += prefixSumExclusive<REORDER_NUMBER_OF_THREADS_PER_BLOCK>( prefix, &localPrefixSum[i] );
	}

	__syncthreads();

	// reorder
	for( int i = 0; i < RADIX_SORT_BLOCK_SIZE; i += REORDER_NUMBER_OF_THREADS_PER_BLOCK )
	{
		uint32_t itemIndex = blockIndex * RADIX_SORT_BLOCK_SIZE + i + threadIdx.x;
		RADIX_SORT_KEY_TYPE item;
		uint32_t bucketIndex;
		if( itemIndex < numberOfInputs )
		{
			item = inputKeys[itemIndex];
			bucketIndex = ( item >> bitLocation ) & 0xFF;
		}
		
		for( int w = 0; w < REORDER_NUMBER_OF_WARPS; w++ )
		{
			for( int i = 0; i < 256; i += REORDER_NUMBER_OF_THREADS_PER_BLOCK )
			{
				matchMasks[w][i + threadIdx.x] = 0;
			}
		}
		__syncthreads();

		int warp = threadIdx.x / 32;
		int lane = threadIdx.x % 32;
		
		if( itemIndex < numberOfInputs )
		{
			atomicOr( &matchMasks[warp][bucketIndex], 1u << lane );
		}

		__syncthreads();

		if( itemIndex < numberOfInputs )
		{
			uint32_t matchMask = matchMasks[warp][bucketIndex];
			uint32_t lowerMask = ( 1u << lane ) - 1;
			uint32_t offset = __popc( matchMask & lowerMask );

			for( int w = 0; w < warp; w++ )
			{
				offset += __popc( matchMasks[w][bucketIndex] );
			}

			uint32_t location = localPrefixSum[bucketIndex];
			elementIndices[location + offset] = i + threadIdx.x;
		}

		__syncthreads();

		if( itemIndex < numberOfInputs )
		{
			atomicInc( &localPrefixSum[bucketIndex], 0xFFFFFFFF );
		}
		__syncthreads();
	}

	// load
	for( int i = 0; i < 256; i += REORDER_NUMBER_OF_THREADS_PER_BLOCK )
	{
		uint32_t counterIndex = ( i + threadIdx.x ) /* bucketIndex */ * numberOfBlocks + blockIndex;
		localPrefixSum[i + threadIdx.x] = sums[counterIndex];
	}
	__syncthreads();

	for( int i = 0; i < RADIX_SORT_BLOCK_SIZE; i += REORDER_NUMBER_OF_THREADS_PER_BLOCK )
	{
		uint32_t itemIndex = blockIndex * RADIX_SORT_BLOCK_SIZE + i + threadIdx.x;
		RADIX_SORT_KEY_TYPE item;
		uint32_t bucketIndex;
		uint32_t srcIndex;
		if( itemIndex < numberOfInputs )
		{
			srcIndex = blockIndex * RADIX_SORT_BLOCK_SIZE + elementIndices[i + threadIdx.x];
			item = inputKeys[srcIndex];
			bucketIndex = ( item >> bitLocation ) & 0xFF;
		}

		for( int w = 0; w < REORDER_NUMBER_OF_WARPS; w++ )
		{
			for( int i = 0; i < 256; i += REORDER_NUMBER_OF_THREADS_PER_BLOCK )
			{
				matchMasks[w][i + threadIdx.x] = 0;
			}
		}
		__syncthreads();

		int warp = threadIdx.x / 32;
		int lane = threadIdx.x % 32;

		if( itemIndex < numberOfInputs )
		{
			atomicOr( &matchMasks[warp][bucketIndex], 1u << lane );
		}

		__syncthreads();

		if( itemIndex < numberOfInputs )
		{
			uint32_t matchMask = matchMasks[warp][bucketIndex];
			uint32_t lowerMask = ( 1u << lane ) - 1;
			uint32_t offset = __popc( matchMask & lowerMask );

			for( int w = 0; w < warp; w++ )
			{
				offset += __popc( matchMasks[w][bucketIndex] );
			}

			uint32_t location = localPrefixSum[bucketIndex];
			outputKeys[location + offset] = item;
			if( keyPair )
			{
				outputValues[location + offset] = inputValues[srcIndex];
			}
		}

		__syncthreads();

		if( itemIndex < numberOfInputs )
		{
			atomicInc( &localPrefixSum[bucketIndex], 0xFFFFFFFF );
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
			bucketIndex = ( item >> bitLocation ) & 0xFF;
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