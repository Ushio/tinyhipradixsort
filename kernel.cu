typedef unsigned long long uint64_t;
typedef unsigned int uint32_t;
typedef unsigned short uint16_t;
typedef unsigned char uint8_t;

#define RADIX_SORT_BLOCK_SIZE 1024

// TODO out of range handling
#define RADIX_SORT_PREFIX_SCAN_BLOCK 4096

//#define RADIX_SORT_TYPE uint64_t
#define RADIX_SORT_TYPE uint32_t

#if defined( CUDART_VERSION ) && CUDART_VERSION >= 9000
	#define ITS 1
#endif

__device__ inline int div_round_up( int val, int divisor )
{
	return ( val + divisor - 1 ) / divisor;
}

extern "C" __global__ void blockCount( RADIX_SORT_TYPE* inputs, uint64_t numberOfInputs, uint32_t* counters, uint32_t bitLocation )
{
	__shared__ uint32_t localCounters[256];
	for( int i = 0; i < 256; i += 32 )
	{
		localCounters[i + threadIdx.x] = 0;
	}

	__syncthreads();

	for( int i = 0; i < RADIX_SORT_BLOCK_SIZE; i += 32 )
	{
		uint64_t itemIndex = blockIdx.x * RADIX_SORT_BLOCK_SIZE + threadIdx.x + i;
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
		int numberOfBlocks = div_round_up( numberOfInputs, RADIX_SORT_BLOCK_SIZE );
		int bucketIndex = i + threadIdx.x;
		int blockIndex = blockIdx.x;
		uint64_t counterIndex = bucketIndex * numberOfBlocks + blockIndex;
		counters[counterIndex] = localCounters[bucketIndex];
	}
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
	*sum = __shfl( x, 31 );
#endif
	*p = x - val;
}


extern "C" __global__ void prefixSumExclusive( uint32_t* inputs, uint64_t numberOfInputs, uint32_t* sums, uint32_t* iterator, uint32_t* globalPrefix )
{
	__shared__ uint32_t localPrefixSum[RADIX_SORT_PREFIX_SCAN_BLOCK];

	int blockIndex = blockIdx.x;
	int prefix = 0;
	for( int i = 0; i < RADIX_SORT_PREFIX_SCAN_BLOCK; i += 32 )
	{
		int itemIndex = blockIndex * RADIX_SORT_PREFIX_SCAN_BLOCK + i + threadIdx.x;
		uint32_t p;
		uint32_t s;
		warpPrefixSumExclusive( itemIndex < numberOfInputs ? inputs[itemIndex] : 0, &p, &s );
		localPrefixSum[i + threadIdx.x] = prefix + p;
		prefix += s;
	}

	int gp;
	if( threadIdx.x == 0 )
	{
		while( atomicAdd( iterator, 0 ) != blockIndex )
			;

		__threadfence();

		gp = *globalPrefix;
		*globalPrefix += prefix;

		__threadfence();

		atomicInc( iterator, 0xFFFFFFFF );
	}

	//  64bit it = ( counter )( offset ) 
	//int gp;
	//if( threadIdx.x == 0 )
	//{
	//	uint64_t old = counter[0];
	//	uint64_t expected;
	//	do
	//	{
	//		uint32_t previous = old & 0xFFFFFFFF;
	//		expected = previous + ( (uint64_t)( blockIndex ) << 32 );
	//		uint64_t newValue = previous + prefix + ( (uint64_t)( blockIndex + 1 ) << 32 );
	//		old = atomicCAS( iterator, expected, newValue );
	//	} while( old != expected );
	//	gp = old & 0xFFFFFFFF;
	//}

	//int gp;
	//if( threadIdx.x == 0 )
	//{
	//	struct Context
	//	{
	//		uint64_t prefix : 32;
	//		uint64_t iterator : 32;
	//	};
	//	Context old = *(Context*)iterator;
	//	Context expected;
	//	do
	//	{
	//		expected.prefix = old.prefix;
	//		expected.iterator = blockIndex;
	//		Context newValue;
	//		newValue.prefix = old.prefix + prefix;
	//		newValue.iterator = blockIndex + 1;
	//		old = atomicCAS( iterator, expected, &newValue );
	//	} while( old != expected );
	//	gp = old & 0xFFFFFFFF;
	//}

#if defined( ITS )
	__syncwarp( 0xffffffff );
	gp = __shfl_sync( 0xffffffff, gp, 0 );
#else
	__syncthreads();
	gp = __shfl( gp, 0 );
#endif

	for( int i = 0; i < RADIX_SORT_PREFIX_SCAN_BLOCK; i += 32 )
	{
		int itemIndex = blockIndex * RADIX_SORT_PREFIX_SCAN_BLOCK + i + threadIdx.x;
		if (itemIndex < numberOfInputs)
		{
			sums[itemIndex] = gp + localPrefixSum[i + threadIdx.x];
		}
	}
}

extern "C" __global__ void reorder( RADIX_SORT_TYPE* inputs, RADIX_SORT_TYPE* outputs, uint64_t numberOfInputs, uint32_t* sums, uint32_t bitLocation )
{
#if 1
	__shared__ uint32_t localPrefixSum[256];
	__shared__ uint16_t elementIndices[RADIX_SORT_BLOCK_SIZE];

	int blockIndex = blockIdx.x;
	int numberOfBlocks = div_round_up( numberOfInputs, RADIX_SORT_BLOCK_SIZE );
	for( int i = 0; i < 256; i += 32 )
	{
		localPrefixSum[i + threadIdx.x] = 0;
	}
	__syncthreads();

	// count
	for( int i = 0; i < RADIX_SORT_BLOCK_SIZE; i += 32 )
	{
		int itemIndex = blockIndex * RADIX_SORT_BLOCK_SIZE + i + threadIdx.x;
		if( itemIndex < numberOfInputs )
		{
			auto item = inputs[itemIndex];
			int bucketIndex = ( item >> bitLocation ) & 0xFF;
			atomicInc( &localPrefixSum[bucketIndex], 0xFFFFFFFF );
		}
	}

	// prefix sum
	int prefix = 0;
	for( int i = 0; i < 256; i += 32 )
	{
		int digits = i + threadIdx.x;
		uint32_t p;
		uint32_t s;
		warpPrefixSumExclusive( localPrefixSum[digits], &p, &s );
		localPrefixSum[digits] = prefix + p;
		prefix += s;
	}

	// reorder
	for( int i = 0; i < RADIX_SORT_BLOCK_SIZE; i += 32 )
	{
		uint64_t itemIndex = (uint64_t)blockIndex * RADIX_SORT_BLOCK_SIZE + i + threadIdx.x;
		RADIX_SORT_TYPE item;
		uint32_t bucketIndex;
		if( itemIndex < numberOfInputs )
		{
			item = inputs[itemIndex];
			bucketIndex = ( item >> bitLocation ) & 0xFF;
		}
		uint32_t location = -1;
		for( int j = 0; j < 32; j++ )
		{
			if( j == threadIdx.x )
			{
				if( itemIndex < numberOfInputs )
				{
					location = localPrefixSum[bucketIndex]++;
				}
			}
			__syncthreads();
		}
		if( location != -1 )
		{
			elementIndices[location] = i + threadIdx.x;
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
		uint64_t itemIndex = (uint64_t)blockIndex * RADIX_SORT_BLOCK_SIZE + i + threadIdx.x;
		RADIX_SORT_TYPE item;
		uint32_t bucketIndex;
		if( itemIndex < numberOfInputs )
		{
			item = inputs[blockIndex * RADIX_SORT_BLOCK_SIZE + elementIndices[i + threadIdx.x]];
			bucketIndex = ( item >> bitLocation ) & 0xFF;
		}

		uint32_t location = -1;
		for( int j = 0; j < 32; j++ )
		{
			if( j == threadIdx.x )
			{
				if( itemIndex < numberOfInputs )
				{
					location = localPrefixSum[bucketIndex]++;
				}
			}
			__syncthreads();
		}
		if( location != -1 )
		{
			outputs[location] = item;
		}
	}
#else
	__shared__ uint32_t psum[256];

	int blockIndex = blockIdx.x;
	int numberOfBlocks = div_round_up( numberOfInputs, RADIX_SORT_BLOCK_SIZE );
	for( int i = 0; i < 256; i += 32 )
	{
		uint32_t counterIndex = ( i + threadIdx.x ) /* bucketIndex */ * numberOfBlocks + blockIndex;
		psum[i + threadIdx.x] = sums[counterIndex];
	}
	__syncthreads();

	for( int i = 0; i < RADIX_SORT_BLOCK_SIZE; i += 32 )
	{
		uint64_t itemIndex = (uint64_t)blockIndex * RADIX_SORT_BLOCK_SIZE + i + threadIdx.x;
		RADIX_SORT_TYPE item;
		uint32_t bucketIndex;
		if( itemIndex < numberOfInputs )
		{
			item = inputs[itemIndex];
			bucketIndex = ( item >> bitLocation ) & 0xFF;
		}

		uint32_t location = -1;
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
		if( location != -1 )
		{
			outputs[location] = item;
		}
	}
#endif
}