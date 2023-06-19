typedef unsigned long long uint64_t;
typedef unsigned int uint32_t;
typedef unsigned short uint16_t;
typedef unsigned char uint8_t;

#define RADIX_SORT_BLOCK_SIZE 1024

// TODO out of range handling
#define RADIX_SORT_PREFIX_SCAN_BLOCK 4096

#if defined( CUDART_VERSION ) && CUDART_VERSION >= 9000
	#define ITS 1
#endif

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

		gp = atomicAdd( globalPrefix, prefix );

		__threadfence();

		atomicInc( iterator, 0xFFFFFFFF );
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
		int itemIndex = blockIndex * RADIX_SORT_PREFIX_SCAN_BLOCK + i + threadIdx.x;
		if (itemIndex < numberOfInputs)
		{
			sums[itemIndex] = gp + localPrefixSum[i + threadIdx.x];
		}
	}
}

extern "C" __global__ void reorder( uint64_t* inputs, uint64_t* outputs, uint64_t numberOfInputs, uint32_t* sums, uint32_t bitLocation )
{
	// Maybe we can break prefix sum idx. 
	int blockIndex = threadIdx.x + blockDim.x * blockIdx.x;
	int numberOfBlocks = div_round_up( numberOfInputs, RADIX_SORT_BLOCK_SIZE );

	//for( int i = 0; i < RADIX_SORT_BLOCK_SIZE; i++ )
	//{
	//	uint64_t itemIndex = (uint64_t)blockIndex * RADIX_SORT_BLOCK_SIZE + i;
	//	if( numberOfInputs <= itemIndex )
	//	{
	//		break;
	//	}
	//	uint64_t item = inputs[itemIndex];
	//	uint32_t bucketIndex = ( item >> bitLocation ) & 0xFF;

	//	uint32_t counterIndex = bucketIndex * numberOfBlocks + blockIndex;
	//	uint32_t location = sums[counterIndex]++;
	//	outputs[location] = item;
	//}

	#define N_BATCH 16

	for( int i = 0; i < RADIX_SORT_BLOCK_SIZE; i += N_BATCH )
	{
		uint64_t items[N_BATCH];
		for( int j = 0; j < N_BATCH; j++ )
		{
			uint64_t itemIndex = (uint64_t)blockIndex * RADIX_SORT_BLOCK_SIZE + i + j;
			if( numberOfInputs <= itemIndex )
			{
				break;
			}
			items[j] = inputs[itemIndex];
		}

		for (int j = 0; j < N_BATCH; j++)
		{
			uint64_t itemIndex = (uint64_t)blockIndex * RADIX_SORT_BLOCK_SIZE + i + j;
			if( numberOfInputs <= itemIndex )
			{
				break;
			}

			uint32_t bucketIndex = ( items[j] >> bitLocation ) & 0xFF;
			uint32_t counterIndex = bucketIndex * numberOfBlocks + blockIndex;
			uint32_t location = sums[counterIndex]++;
			outputs[location] = items[j];
		}
	}
}