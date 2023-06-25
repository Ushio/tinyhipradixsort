typedef unsigned long long uint64_t;
typedef unsigned int uint32_t;
typedef unsigned short uint16_t;
typedef unsigned char uint8_t;

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
	clearShared<256, 32, uint32_t>( localCounters, 0 );
	__syncthreads();

	for( int i = 0; i < RADIX_SORT_BLOCK_SIZE; i += 32 )
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

	for( int i = 0; i < 256; i += 32 )
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

__device__ void reorder( RADIX_SORT_KEY_TYPE* inputKeys, RADIX_SORT_KEY_TYPE* outputKeys, RADIX_SORT_VALUE_TYPE* inputValues, RADIX_SORT_VALUE_TYPE* outputValues, bool keyPair, uint32_t numberOfInputs, uint32_t* sums, uint32_t bitLocation )
{
#if 1
	__shared__ uint32_t localPrefixSum[256];
	__shared__ uint16_t elementIndices[RADIX_SORT_BLOCK_SIZE];
	__shared__ uint32_t matchMasks[256];

	uint32_t blockIndex = blockIdx.x;
	uint32_t numberOfBlocks = div_round_up( numberOfInputs, RADIX_SORT_BLOCK_SIZE );
	clearShared<256, 32, uint32_t>( localPrefixSum, 0 );
	__syncthreads();

	// count
	for( int i = 0; i < RADIX_SORT_BLOCK_SIZE; i += 32 )
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
	uint16_t prefix = 0;
	for( int i = 0; i < 256; i += 32 )
	{
		int digits = i + threadIdx.x;
		uint32_t p;
		uint32_t s;
		warpPrefixSumExclusive( localPrefixSum[digits], &p, &s );

		__syncthreads();

		localPrefixSum[digits] = prefix + p;
		prefix += s;
	}

	__syncthreads();

	// reorder
	for( int i = 0; i < RADIX_SORT_BLOCK_SIZE; i += 32 )
	{
		uint32_t itemIndex = blockIndex * RADIX_SORT_BLOCK_SIZE + i + threadIdx.x;
		RADIX_SORT_KEY_TYPE item;
		uint32_t bucketIndex;
		if( itemIndex < numberOfInputs )
		{
			item = inputKeys[itemIndex];
			bucketIndex = ( item >> bitLocation ) & 0xFF;
		}

		clearShared<256, 32, uint32_t>( matchMasks, 0 );
		__syncthreads();
		if( itemIndex < numberOfInputs )
		{
			atomicOr( &matchMasks[bucketIndex], 1u << threadIdx.x );
		}
		__syncthreads();

		if( itemIndex < numberOfInputs )
		{
			uint32_t matchMask = matchMasks[bucketIndex];
			uint32_t lowerMask = ( 1u << threadIdx.x ) - 1;
			uint32_t offset = __popc( matchMask & lowerMask );
			uint32_t location = localPrefixSum[bucketIndex];
			elementIndices[location + offset] = i + threadIdx.x;
		}

		__syncthreads();

		if( itemIndex < numberOfInputs )
		{
			atomicInc( &localPrefixSum[bucketIndex], 0xFFFFFFFF );
		}
	}

	// load
	for( int i = 0; i < 256; i += 32 )
	{
		uint32_t counterIndex = ( i + threadIdx.x ) /* bucketIndex */ * numberOfBlocks + blockIndex;
		localPrefixSum[i + threadIdx.x] = sums[counterIndex];
	}
	__syncthreads();

	for( int i = 0; i < RADIX_SORT_BLOCK_SIZE; i += 32 )
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

		clearShared<256, 32, uint32_t>( matchMasks, 0 );
		__syncthreads();
		if( itemIndex < numberOfInputs )
		{
			atomicOr( &matchMasks[bucketIndex], 1u << threadIdx.x );
		}
		__syncthreads();

		if( itemIndex < numberOfInputs )
		{
			uint32_t matchMask = matchMasks[bucketIndex];
			uint32_t lowerMask = ( 1u << threadIdx.x ) - 1;
			uint32_t offset = __popc( matchMask & lowerMask );
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
	for( int i = 0; i < 256; i += 32 )
	{
		uint32_t counterIndex = ( i + threadIdx.x ) /* bucketIndex */ * numberOfBlocks + blockIndex;
		psum[i + threadIdx.x] = sums[counterIndex];
	}
	__syncthreads();

	for( int i = 0; i < RADIX_SORT_BLOCK_SIZE; i += 32 )
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
		for( int j = 0; j < 32; j++ )
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