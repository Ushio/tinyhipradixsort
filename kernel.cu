typedef unsigned long long uint64_t;
typedef unsigned int uint32_t;
typedef unsigned short uint16_t;
typedef unsigned char uint8_t;

#define RADIX_SORT_BLOCK_SIZE 1024
#define RADIX_SORT_PREFIX_SCAN_BLOCK 256

__device__ inline int div_round_up( int val, int divisor )
{
	return ( val + divisor - 1 ) / divisor;
}

extern "C" __global__ void blockCount( uint64_t* inputs, uint64_t numberOfInputs, uint32_t* counters, uint32_t bitLocation )
{
	__shared__ uint32_t localCounters[256];
	if( threadIdx.x < 256 )
	{
		localCounters[threadIdx.x] = 0;
	}
	__syncthreads();

	uint64_t itemIndex = blockIdx.x * blockDim.x + threadIdx.x;
	if( itemIndex < numberOfInputs )
	{
		uint64_t item = inputs[itemIndex];
		uint32_t bits = ( item >> bitLocation ) & 0xFF;
		atomicInc( &localCounters[bits], 0xFFFFFFFF );
	}

	__syncthreads();

	if( threadIdx.x < 256 )
	{
		// column major: 
		int numberOfBlocks = div_round_up( numberOfInputs, RADIX_SORT_BLOCK_SIZE );
		int bucketIndex = threadIdx.x;
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
		__syncwarp( 0xffffffff );
		uint32_t y = __shfl_up_sync( 0xffffffff, x, offset );

		if( offset <= threadIdx.x )
		{
			x += y;
		}
	}

	__syncwarp( 0xffffffff );
	*sum = __shfl_sync( 0xffffffff, x, 31 );
	*p = x - val;
}


extern "C" __global__ void prefixSumExclusive( uint32_t* inputs, uint64_t numberOfInputs, uint32_t* sums, uint32_t* iterator, uint32_t* globalPrefix )
{
	//uint32_t r;
	//uint32_t s;
	//warpPrefixSumExclusive( 1, &r, &s );
	//printf( "[%d] %d %d\n", threadIdx.x, r, s );

	//assert( 0 );
	// printf( "%d %d\n", threadIdx.x, __shfl_up_sync( 0xffffffff, threadIdx.x, 1 ) );
	// printf( "%d %d\n", threadIdx.x, warpPrefixSum( 2 ) );

	__shared__ uint32_t localPrefixSum[RADIX_SORT_PREFIX_SCAN_BLOCK];

	int blockIndex = blockIdx.x;
	int prefix = 0;
	for( int i = 0; i < RADIX_SORT_PREFIX_SCAN_BLOCK; i += 32 )
	{
		int itemIndex = blockIndex * RADIX_SORT_PREFIX_SCAN_BLOCK + i + threadIdx.x;
		uint32_t p;
		uint32_t s;
		warpPrefixSumExclusive( inputs[itemIndex], &p, &s );
		localPrefixSum[i + threadIdx.x] = prefix + p;
		prefix += s;
	}

	int gp;
	if( threadIdx.x == 0 )
	{
		while( atomicAdd( iterator, 0 ) != blockIndex )
			;

		__threadfence();

		gp = atomicAdd( globalPrefix, prefix );

		__threadfence();

		atomicInc( iterator, 0xFFFFFFFF );
	}

	__syncwarp( 0xffffffff );
	gp = __shfl_sync( 0xffffffff, gp, 0 );

	for( int i = 0; i < RADIX_SORT_PREFIX_SCAN_BLOCK; i += 32 )
	{
		int itemIndex = blockIndex * RADIX_SORT_PREFIX_SCAN_BLOCK + i + threadIdx.x;
		sums[itemIndex] = gp + localPrefixSum[i + threadIdx.x];
	}
}

