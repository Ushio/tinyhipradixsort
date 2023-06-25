#pragma once

#include <string>
#include <vector>
#include <memory>
#include <map>
#include <inttypes.h>
#include <Orochi/Orochi.h>

#define ARG_DEBINFO_NV "--generate-line-info"
#define ARG_DEBINFO_AMD "-g"

#include <intrin.h>
#define THRS_ASSERT(ExpectTrue) if((ExpectTrue) == 0) { __debugbreak(); }

//#define THRS_KERNEL_FROM_FILE 1

namespace thrs
{
const char* theKernel = R"(
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

            orortcResult re;
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
        U64
    };
    enum class ValueType
    {
        U32,
        U64,
        U128,
    };
	inline int bytesOf( KeyType valueType )
	{
		switch( valueType )
		{
		case KeyType::U32:
			return 4;
		case KeyType::U64:
			return 8;
		}
		THRS_ASSERT( 0 );
		return 0;
	}
	inline int bytesOf( ValueType valueType )
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


    inline uint32_t div_round_up( uint32_t val, uint32_t divisor )
	{
		return ( val + divisor - 1 ) / divisor;
	}
	inline uint32_t next_multiple( uint32_t val, uint32_t divisor )
	{
		return div_round_up( val, divisor ) * divisor;
	}

    class RadixSort
    {
    public:
        struct Config
        {
            int radixSortBlockSize = 2048;
            int prefixScanBlockSize = 8192;
            KeyType keyType = KeyType::U32;
            ValueType valueType = ValueType::U32;

			template <class KEY>
			void configureWithKey()
			{
				static_assert( sizeof( KEY ) == 4 || sizeof( KEY ) == 8, "" );

				switch( sizeof( KEY ) )
				{
				case 4:
					keyType = KeyType::U32;
					break;
				case 8:
					keyType = KeyType::U64;
					break;
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
			THRS_ASSERT( 32 <= config.radixSortBlockSize );
			THRS_ASSERT( ( config.radixSortBlockSize % 32 ) == 0 );
			THRS_ASSERT( 32 <= config.prefixScanBlockSize );
			THRS_ASSERT( ( config.prefixScanBlockSize % 32 ) == 0 );

            // shader settings
			extraArgs.push_back( std::string( "-DRADIX_SORT_BLOCK_SIZE=" ) + std::to_string( m_config.radixSortBlockSize ) );
			extraArgs.push_back( std::string( "-DRADIX_SORT_PREFIX_SCAN_BLOCK=" ) + std::to_string( m_config.prefixScanBlockSize ) );

            switch( m_config.keyType )
            {
			case KeyType::U32:
				extraArgs.push_back( std::string( "-DRADIX_SORT_KEY_TYPE=uint32_t" ));
				break;
            case KeyType::U64:
				extraArgs.push_back( std::string( "-DRADIX_SORT_KEY_TYPE=uint64_t" ) );
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
			uint64_t keyOutBuffer;
			uint64_t valueOutBuffer;

			uint64_t getTemporaryBufferBytesForSortKeys() const
			{
				return pSumBuffer + keyOutBuffer;
			}
			uint64_t getTemporaryBufferBytesForSortPairs() const
			{
				return pSumBuffer + keyOutBuffer + valueOutBuffer;
			}
			void* getPSumBuffer( void* p ) const
			{
				return p;
			}
			void* getOutputKeyBuffer( void* p ) const
			{
				return (void*)( (uint8_t*)p + pSumBuffer );
			}
			void* getOutputValueBuffer( void* p ) const
			{
				return (void*)( (uint8_t*)p + pSumBuffer + keyOutBuffer );
			}
		};
		TemporaryBufferDef getTemporaryBufferBytes( uint32_t numberOfMaxInputs ) const
        {
			const int alignment = 16;

			TemporaryBufferDef def = {};
			uint64_t numberOfBlocks = div_round_up( numberOfMaxInputs, m_config.radixSortBlockSize );
			def.pSumBuffer = next_multiple( sizeof( uint32_t ) * 256 * numberOfBlocks, alignment );
			def.keyOutBuffer = next_multiple( bytesOf( m_config.keyType ) * numberOfMaxInputs, alignment );
			def.valueOutBuffer = next_multiple( bytesOf( m_config.valueType ) * numberOfMaxInputs, alignment );
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
			void* outputKeyBuffer = def.getOutputKeyBuffer( temporaryBuffer );
			void* outputValueBuffer = def.getOutputValueBuffer( temporaryBuffer );

			uint32_t numberOfBlocks = div_round_up( numberOfInputs, m_config.radixSortBlockSize );

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
					m_shader->launch( "blockCount", args, numberOfBlocks, 1, 1, 32, 1, 1, stream );
				}
				// Prefix Sum
				{
					// OroStopwatch oroStream( stream );
					// oroStream.start();

					ShaderArgument args;
					args.add( pSumBuffer );
					args.add( numberOfBlocks * 256 );
					m_shader->launch( "prefixSumExclusiveInplace", args, div_round_up( numberOfBlocks * 256, m_config.prefixScanBlockSize ), 1, 1, 32, 1, 1, stream );

					// oroStream.stop();
					// float ms = oroStream.getMs();
					// oroStreamSynchronize( stream );

					// printf( "pSum %f ms\n", ms );
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
						m_shader->launch( "reorderKeyPair", args, numberOfBlocks, 1, 1, 32, 1, 1, stream );
					}
					else
					{
						ShaderArgument args;
						args.add( inputKeyBuffer );
						args.add( outputKeyBuffer );
						args.add( numberOfInputs );
						args.add( pSumBuffer );
						args.add( bitLocation );
						m_shader->launch( "reorderKey", args, numberOfBlocks, 1, 1, 32, 1, 1, stream );
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
				oroMemcpyDtoD( (oroDeviceptr)outputKeyBuffer, (oroDeviceptr)inputKeyBuffer, (uint64_t)bytesOf( m_config.keyType ) * numberOfInputs );
				if( keyPair )
				{
					oroMemcpyDtoD( (oroDeviceptr)outputValueBuffer, (oroDeviceptr)inputValueBuffer, (uint64_t)bytesOf( m_config.valueType ) * numberOfInputs );
				}
			}
		}
    private:
        std::unique_ptr<Shader> m_shader;
        Config m_config;
    };
}