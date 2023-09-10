#pragma once

#include <string>
#include <vector>
#include <memory>
#include <map>
#include <inttypes.h>
#include <Orochi/Orochi.h>
//#include <Orochi/OrochiUtils.h>

#define ARG_DEBINFO_NV "--generate-line-info"
#define ARG_DEBINFO_AMD "-g"

#include <intrin.h>
#define THRS_ASSERT(ExpectTrue) if((ExpectTrue) == 0) { __debugbreak(); }

//#define THRS_KERNEL_FROM_FILE 1

#define RADIX_SORT_BLOCK_SIZE 2048
#define RADIX_SORT_PREFIX_SCAN_BLOCK 4096

#define BLOCK_COUNT_NUMBER_OF_WARPS 8
#define BLOCK_COUNT_NUMBER_OF_THREADS_PER_BLOCK ( 32 * BLOCK_COUNT_NUMBER_OF_WARPS )

#define PSUM_NUMBER_OF_WARPS 4
#define PSUM_NUMBER_OF_THREADS_PER_BLOCK ( 32 * PSUM_NUMBER_OF_WARPS )

#define REORDER_NUMBER_OF_WARPS 8
#define REORDER_NUMBER_OF_THREADS_PER_BLOCK ( 32 * REORDER_NUMBER_OF_WARPS )

struct PartitionID
{
	uint32_t flag;
	uint32_t aggregate;
	uint32_t incPrefix;
	uint32_t _pad;
};

// some checks for simplicity
static_assert( BLOCK_COUNT_NUMBER_OF_THREADS_PER_BLOCK <= 256, "please check counters, etc" );
static_assert( ( RADIX_SORT_BLOCK_SIZE % BLOCK_COUNT_NUMBER_OF_THREADS_PER_BLOCK ) == 0, "you may need some adjustments on blockCount" );

static_assert( PSUM_NUMBER_OF_THREADS_PER_BLOCK <= 1024, "" );
static_assert( ( RADIX_SORT_PREFIX_SCAN_BLOCK % PSUM_NUMBER_OF_THREADS_PER_BLOCK ) == 0, "you may need some adjustments on prefixSumExclusiveInplace" );

static_assert( REORDER_NUMBER_OF_THREADS_PER_BLOCK <= 256, "please check prefixSumExclusive, etc" );
static_assert( ( RADIX_SORT_BLOCK_SIZE % ( REORDER_NUMBER_OF_THREADS_PER_BLOCK * 4 ) ) == 0, "you may need some adjustments on reorder" );

static_assert( RADIX_SORT_BLOCK_SIZE <= 4096 /* 2^12 */, "please check ElementLocation" );


namespace thrs
{
const char* theKernel = R"(
typedef unsigned long long uint64_t;
typedef unsigned int uint32_t;
typedef unsigned short uint16_t;
typedef unsigned char uint8_t;

#define RADIX_SORT_BLOCK_SIZE 2048
#define RADIX_SORT_PREFIX_SCAN_BLOCK 4096

#define BLOCK_COUNT_NUMBER_OF_WARPS 8
#define BLOCK_COUNT_NUMBER_OF_THREADS_PER_BLOCK ( 32 * BLOCK_COUNT_NUMBER_OF_WARPS )

#define PSUM_NUMBER_OF_WARPS 4
#define PSUM_NUMBER_OF_THREADS_PER_BLOCK ( 32 * PSUM_NUMBER_OF_WARPS )

#define REORDER_NUMBER_OF_WARPS 8
#define REORDER_NUMBER_OF_THREADS_PER_BLOCK ( 32 * REORDER_NUMBER_OF_WARPS )

struct PartitionID
{
	uint32_t flag;
	uint32_t aggregate;
	uint32_t incPrefix;
	uint32_t _pad;
};

enum
{
	PARTITIOIN_FLAG_X = 0,
	PARTITIOIN_FLAG_A,
	PARTITIOIN_FLAG_P,
};

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

extern "C" __global__ void prefixSumExclusiveInplace( uint32_t* inout, uint32_t numberOfInputs, PartitionID* partitions )
{
	__shared__ uint32_t gp;
	__shared__ uint32_t smem[PSUM_NUMBER_OF_THREADS_PER_BLOCK];

	int blockIndex = blockIdx.x;

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
		uint32_t aggregate = smem[0];
		partitions[blockIndex].aggregate = aggregate;

		__threadfence();

		atomicExch( &partitions[blockIndex].flag, PARTITIOIN_FLAG_A );
		
		uint32_t p = 0;
		for( int i = blockIndex - 1 ; 0 <= i ; i-- )
		{
			uint32_t flag;
			do
			{
				flag = atomicAdd( &partitions[i].flag, 0 );
			}
			while( flag == PARTITIOIN_FLAG_X );

			__threadfence();

			if( flag == PARTITIOIN_FLAG_A )
			{
				p += partitions[i].aggregate;
			}
			else
			{
				p += partitions[i].incPrefix;
				break;
			}
		}

		partitions[blockIndex].incPrefix = p + aggregate;

		__threadfence();

		atomicExch( &partitions[blockIndex].flag, PARTITIOIN_FLAG_P );

		gp = p;
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
)";

    inline void loadFileAsVector( std::vector<char>* buffer, const char* fllePath )
    {
	    FILE* fp = fopen( fllePath, "rb" );
	    if( fp == nullptr )
	    {
		    return;
	    }

	    fseek( fp, 0, SEEK_END );

	    buffer->resize( ftell( fp ) );

	    fseek( fp, 0, SEEK_SET );

	    size_t s = fread( buffer->data(), 1, buffer->size(), fp );
	    if( s != buffer->size() )
	    {
		    buffer->clear();
		    return;
	    }
	    fclose( fp );
	    fp = nullptr;
    }
	class Buffer
	{
	public:
		Buffer( const Buffer& ) = delete;
		void operator=( const Buffer& ) = delete;

		Buffer( int64_t bytes )
			: m_bytes( std::max( bytes, 1LL ) )
		{
			oroMalloc( &m_ptr, m_bytes );
		}
		~Buffer()
		{
			oroFree( m_ptr );
		}
		int64_t bytes() const
		{
			return m_bytes;
		}
		char* data()
		{
			return (char*)m_ptr;
		}

	private:
		int64_t m_bytes;
		oroDeviceptr m_ptr;
	};

    struct ShaderArgument
    {
	    template <class T>
	    void add( T p )
	    {
		    int bytes = sizeof( p );
		    int location = m_buffer.size();
		    m_buffer.resize( m_buffer.size() + bytes );
		    memcpy( m_buffer.data() + location, &p, bytes );
		    m_locations.push_back( location );
	    }
	    void clear()
	    {
		    m_buffer.clear();
		    m_locations.clear();
	    }

	    std::vector<void*> kernelParams() const
	    {
		    std::vector<void*> ps;
		    for( int i = 0; i < m_locations.size(); i++ )
		    {
			    ps.push_back( (void*)( m_buffer.data() + m_locations[i] ) );
		    }
		    return ps;
	    }

    private:
	    std::vector<char> m_buffer;
	    std::vector<int> m_locations;
    };
    class Shader
    {
    public:
        Shader( const char* src, const char* kernelLabel, const std::vector<std::string>& extraArgs )
        {
            orortcProgram program = 0;
            orortcCreateProgram( &program, src, kernelLabel, 0, 0, 0 );
            std::vector<std::string> options;

            for( int i = 0; i < extraArgs.size(); ++i )
            {
                options.push_back( extraArgs[i] );
            }

            std::vector<const char*> optionChars;
            for( int i = 0; i < options.size(); ++i )
            {
                optionChars.push_back( options[i].c_str() );
            }

            orortcResult compileResult = orortcCompileProgram( program, optionChars.size(), optionChars.data() );

            size_t logSize = 0;
            orortcGetProgramLogSize( program, &logSize );
            if( 1 < logSize )
            {
                std::vector<char> compileLog( logSize );
                orortcGetProgramLog( program, compileLog.data() );
                printf( "%s", compileLog.data() );
            }
            THRS_ASSERT( compileResult == ORORTC_SUCCESS );

            size_t codeSize = 0;
            orortcGetCodeSize( program, &codeSize );

            m_shaderBinary.resize( codeSize );
            orortcGetCode( program, m_shaderBinary.data() );

            // FILE* fp = fopen( "shader.bin", "wb" );
            // fwrite( m_shaderBinary.data(), m_shaderBinary.size(), 1, fp );
            // fclose( fp );

            orortcDestroyProgram( &program );

			oroError e = oroModuleLoadData( &m_module, m_shaderBinary.data() );
            THRS_ASSERT( e == oroSuccess );
        }
        ~Shader()
        {
            oroModuleUnload( m_module );
        }
        void launch( const char* name,
                        const ShaderArgument& arguments,
                        unsigned int gridDimX, unsigned int gridDimY, unsigned int gridDimZ,
                        unsigned int blockDimX, unsigned int blockDimY, unsigned int blockDimZ,
                        oroStream hStream )
        {
            if( m_functions.count( name ) == 0 )
            {
                oroFunction f = 0;
                oroError e = oroModuleGetFunction( &f, m_module, name );
                THRS_ASSERT( e == oroSuccess );
                m_functions[name] = f;
            }

            auto params = arguments.kernelParams();
            oroFunction f = m_functions[name];
            oroError e = oroModuleLaunchKernel( f, gridDimX, gridDimY, gridDimZ, blockDimX, blockDimY, blockDimZ, 0, hStream, params.data(), 0 );
            THRS_ASSERT( e == oroSuccess );
        }

    private:
        oroModule m_module = 0;
        std::map<std::string, oroFunction> m_functions;
        std::vector<char> m_shaderBinary;
    };

    enum class KeyType
    {
		U32,
		U64,
		F32,
		F64,
    };
    enum class ValueType
    {
        U32,
        U64,
        U128,
    };
	inline uint64_t bytesOf( KeyType valueType )
	{
		switch( valueType )
		{
		case KeyType::U32:
		case KeyType::F32:
			return 4;
		case KeyType::U64:
		case KeyType::F64:
			return 8;
		}
		THRS_ASSERT( 0 );
		return 0;
	}
	inline uint64_t bytesOf( ValueType valueType )
	{
		switch( valueType )
		{
		case ValueType::U32:
			return 4;
		case ValueType::U64:
			return 8;
		case ValueType::U128:
			return 16;
		}
		THRS_ASSERT( 0 );
		return 0;
	}
	enum class SortOrder
	{
		Ascending,
		Descending
	};

    inline uint64_t div_round_up64( uint64_t val, uint64_t divisor )
	{
		return ( val + divisor - 1 ) / divisor;
	}
	inline uint64_t next_multiple64( uint64_t val, uint64_t divisor )
	{
		return div_round_up64( val, divisor ) * divisor;
	}

    class RadixSort
    {
    public:
        struct Config
        {
			// strongly recommended
			bool keyIs16byteAligned = true;

            KeyType keyType = KeyType::U32;
            ValueType valueType = ValueType::U32;

			SortOrder sortOrder = SortOrder::Ascending;

			template <class KEY>
			void configureWithKey()
			{
				static_assert( sizeof( KEY ) == 4 || sizeof( KEY ) == 8, "" );

				if( std::is_same<KEY, float>::value )
				{
					keyType = KeyType::F32;
				}
				else if( std::is_same<KEY, double>::value )
				{
					keyType = KeyType::F64;
				}
				else if( sizeof( KEY ) == 4 )
				{
					keyType = KeyType::U32;
				}
				else if( sizeof( KEY ) == 8 )
				{
					keyType = KeyType::U64;
				}
			}
			template <class KEY, class VALUE>
			void configureWithKeyPair()
			{
				configureWithKey<KEY>();

				static_assert( sizeof( VALUE ) == 4 || sizeof( VALUE ) == 8 || sizeof( VALUE ) == 16, "" );

				switch( sizeof( VALUE ) )
				{
				case 4:
					valueType = ValueType::U32;
					break;
				case 8:
					valueType = ValueType::U64;
					break;
				case 16:
					valueType = ValueType::U128;
					break;
				}
			}
        };

		RadixSort( std::vector<std::string> extraArgs, const Config& config = Config() )
        :m_config( config )
        {
            switch( m_config.keyType )
            {
			case KeyType::U32:
				extraArgs.push_back( std::string( "-DRADIX_SORT_KEY_TYPE=uint32_t" ) );
				break;
			case KeyType::U64:
				extraArgs.push_back( std::string( "-DRADIX_SORT_KEY_TYPE=uint64_t" ) );
				break;
			case KeyType::F32:
				extraArgs.push_back( std::string( "-DRADIX_SORT_KEY_TYPE=float" ) );
				break;
			case KeyType::F64:
				extraArgs.push_back( std::string( "-DRADIX_SORT_KEY_TYPE=double" ) );
				break;
            }

            switch( m_config.valueType )
			{
			case ValueType::U32:
				extraArgs.push_back( std::string( "-DRADIX_SORT_VALUE_TYPE=uint32_t" ) );
				break;
			case ValueType::U64:
				extraArgs.push_back( std::string( "-DRADIX_SORT_VALUE_TYPE=uint64_t" ) );
				break;
			case ValueType::U128:
				extraArgs.push_back( std::string( "-DRADIX_SORT_VALUE_TYPE=uint4" ) );
				break;
			}

			if( m_config.keyIs16byteAligned )
			{
				extraArgs.push_back( std::string( "-DKEY_IS_16BYTE_ALIGNED=1" ) );
			}

			if( m_config.sortOrder == SortOrder::Descending )
			{
				extraArgs.push_back( std::string( "-DDESCENDING_ORDER=1" ) );
			}

#if defined( THRS_KERNEL_FROM_FILE )
			std::vector<char> src;
			loadFileAsVector( &src, "../kernel.cu" );
			THRS_ASSERT( 0 < src.size() );
			src.push_back( '\0' );
			m_shader = std::unique_ptr<Shader>( new Shader( src.data(), "kernel.cu", extraArgs ) );
#else
			m_shader = std::unique_ptr<Shader>( new Shader( theKernel, "kernel.cu", extraArgs ) );
#endif

			
        }

		struct TemporaryBufferDef
		{
			uint64_t pSumBuffer;
			uint64_t dlbBuffer;
			uint64_t keyOutBuffer;
			uint64_t valueOutBuffer;

			uint64_t getTemporaryBufferBytesForSortKeys() const
			{
				return pSumBuffer + dlbBuffer + keyOutBuffer;
			}
			uint64_t getTemporaryBufferBytesForSortPairs() const
			{
				return pSumBuffer + dlbBuffer + keyOutBuffer + valueOutBuffer;
			}
			void* getPSumBuffer( void* p ) const
			{
				return p;
			}
			void* getDLBBuffer( void* p ) const
			{
				return (void*)( (uint8_t*)p + pSumBuffer );
			}
			void* getOutputKeyBuffer( void* p ) const
			{
				return (void*)( (uint8_t*)p + pSumBuffer + dlbBuffer );
			}
			void* getOutputValueBuffer( void* p ) const
			{
				return (void*)( (uint8_t*)p + pSumBuffer + dlbBuffer + keyOutBuffer );
			}
		};
		TemporaryBufferDef getTemporaryBufferBytes( uint32_t numberOfMaxInputs ) const
        {
			const int alignment = 16;

			TemporaryBufferDef def = {};
			uint64_t numberOfBlocks = div_round_up64( numberOfMaxInputs, RADIX_SORT_BLOCK_SIZE );
			def.pSumBuffer = next_multiple64( sizeof( uint32_t ) * 256 * numberOfBlocks, alignment );
			int nPartitions = div_round_up64( numberOfBlocks * 256, RADIX_SORT_PREFIX_SCAN_BLOCK );
			def.dlbBuffer = next_multiple64( sizeof( PartitionID ) * nPartitions, alignment );
			def.keyOutBuffer = next_multiple64( bytesOf( m_config.keyType ) * numberOfMaxInputs, alignment );
			def.valueOutBuffer = next_multiple64( bytesOf( m_config.valueType ) * numberOfMaxInputs, alignment );
			return def;
        }

		void sortKeys( void* inputKeyBuffer, uint32_t numberOfInputs, void* temporaryBuffer, int startBits, int endBits, oroStream stream )
        {
			sort( inputKeyBuffer, nullptr, false, numberOfInputs, temporaryBuffer, startBits, endBits, stream );
        }
		void sortPairs( void* inputKeyBuffer, void* inputValueBuffer, uint32_t numberOfInputs, void* temporaryBuffer, int startBits, int endBits, oroStream stream )
		{
			sort( inputKeyBuffer, inputValueBuffer, true, numberOfInputs, temporaryBuffer, startBits, endBits, stream );
		}
	private:
		void sort( void* inputKeyBuffer, void* inputValueBuffer, bool keyPair, uint32_t numberOfInputs, void* temporaryBuffer, int startBits, int endBits, oroStream stream )
		{
			THRS_ASSERT( ( ( endBits - startBits ) % 8 ) == 0 );

			// Buffers
			TemporaryBufferDef def = getTemporaryBufferBytes( numberOfInputs );
			void* pSumBuffer = def.getPSumBuffer( temporaryBuffer );
			void* dlbBuffer = def.getDLBBuffer( temporaryBuffer );
			void* outputKeyBuffer = def.getOutputKeyBuffer( temporaryBuffer );
			void* outputValueBuffer = def.getOutputValueBuffer( temporaryBuffer );

			uint32_t numberOfBlocks = div_round_up64( numberOfInputs, RADIX_SORT_BLOCK_SIZE );

			int iteration = 0;
			for( int i = 0; ( startBits + i * 8 ) < endBits; i++ )
			{
				uint32_t bitLocation = startBits + i * 8;

				// counter
				{
					ShaderArgument args;
					args.add( inputKeyBuffer );
					args.add( numberOfInputs );
					args.add( pSumBuffer );
					args.add( bitLocation );
					m_shader->launch( "blockCount", args, numberOfBlocks, 1, 1, BLOCK_COUNT_NUMBER_OF_THREADS_PER_BLOCK, 1, 1, stream );
				}
				// Prefix Sum
				{
					//OroStopwatch oroStream( stream );
					//oroStream.start();
					int nPartitions = div_round_up64( numberOfBlocks * 256, RADIX_SORT_PREFIX_SCAN_BLOCK );
					oroMemsetD8Async( (oroDeviceptr)dlbBuffer, 0, sizeof( PartitionID ) * nPartitions, stream );
					
					ShaderArgument args;
					args.add( pSumBuffer );
					args.add( numberOfBlocks * 256 );
					args.add( dlbBuffer );
					m_shader->launch( "prefixSumExclusiveInplace", args, div_round_up64( numberOfBlocks * 256, RADIX_SORT_PREFIX_SCAN_BLOCK ), 1, 1, PSUM_NUMBER_OF_THREADS_PER_BLOCK, 1, 1, stream );
				
					//oroStream.stop();
					//float ms = oroStream.getMs();
					//oroStreamSynchronize( stream );
					//
					//printf( "psum %f ms\n", ms );
				}
				// reorder
				{
					// OroStopwatch oroStream( stream );
					// oroStream.start();

					if( keyPair )
					{
						ShaderArgument args;
						args.add( inputKeyBuffer );
						args.add( outputKeyBuffer );
						args.add( inputValueBuffer );
						args.add( outputValueBuffer );
						args.add( numberOfInputs );
						args.add( pSumBuffer );
						args.add( bitLocation );
						m_shader->launch( "reorderKeyPair", args, numberOfBlocks, 1, 1, REORDER_NUMBER_OF_THREADS_PER_BLOCK, 1, 1, stream );
					}
					else
					{
						ShaderArgument args;
						args.add( inputKeyBuffer );
						args.add( outputKeyBuffer );
						args.add( numberOfInputs );
						args.add( pSumBuffer );
						args.add( bitLocation );
						m_shader->launch( "reorderKey", args, numberOfBlocks, 1, 1, REORDER_NUMBER_OF_THREADS_PER_BLOCK, 1, 1, stream );
					}

					// oroStream.stop();
					// float ms = oroStream.getMs();
					// oroStreamSynchronize( stream );

					// printf( "reorder %f ms\n", ms );

					iteration++;
				}
				std::swap( inputKeyBuffer, outputKeyBuffer );
				std::swap( inputValueBuffer, outputValueBuffer );
			}

			if( (iteration % 2) == 1 )
			{
				oroMemcpyDtoD( (oroDeviceptr)outputKeyBuffer, (oroDeviceptr)inputKeyBuffer, bytesOf( m_config.keyType ) * numberOfInputs );
				if( keyPair )
				{
					oroMemcpyDtoD( (oroDeviceptr)outputValueBuffer, (oroDeviceptr)inputValueBuffer, bytesOf( m_config.valueType ) * numberOfInputs );
				}
			}
		}
    private:
        std::unique_ptr<Shader> m_shader;
        Config m_config;
    };
}