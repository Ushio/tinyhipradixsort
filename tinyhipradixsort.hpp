#pragma once

#include <string>
#include <vector>
#include <memory>
#include <inttypes.h>
#include <Orochi/Orochi.h>

#define ARG_DEBINFO_NV "--generate-line-info"
#define ARG_DEBINFO_AMD "-g"

#include <intrin.h>
#define THRS_ASSERT(ExpectTrue) if((ExpectTrue) == 0) { __debugbreak(); }

namespace thrs
{
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
        Shader( const char* filename, const char* kernelLabel, const std::vector<std::string>& extraArgs )
        {
            std::vector<char> src;
			loadFileAsVector( &src, filename );
            THRS_ASSERT( 0 < src.size() );
            src.push_back( '\0' );

            orortcProgram program = 0;
            orortcCreateProgram( &program, src.data(), kernelLabel, 0, 0, 0 );
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
			m_shader = std::unique_ptr<Shader>( new Shader( "../kernel.cu", "kernel.cu", extraArgs ) );
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
		};
		TemporaryBufferDef getTemporaryBufferBytes( uint32_t numberOfMaxInputs ) const
        {
			const int alignment = 16;

			TemporaryBufferDef def = {};
			uint64_t numberOfBlocks = div_round_up( numberOfMaxInputs, m_config.radixSortBlockSize );
			def.pSumBuffer = next_multiple( sizeof( uint32_t ) * 256 * numberOfBlocks, alignment );

			uint64_t keyBytes = m_config.keyType == KeyType::U32 ? sizeof( uint32_t ) : sizeof( uint64_t );
			def.keyOutBuffer = next_multiple( keyBytes * numberOfMaxInputs, alignment );

			uint64_t valueBytes = 0;
			switch( m_config.valueType )
			{
			case ValueType::U32:
				valueBytes = sizeof( uint32_t );
				break;
			case ValueType::U64:
				valueBytes = sizeof( uint32_t ) * 2;
				break;
			case ValueType::U128:
				valueBytes = sizeof( uint32_t ) * 4;
				break;
			}
			def.valueOutBuffer = next_multiple( keyBytes * numberOfMaxInputs, alignment );
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
			void* pSumBuffer = temporaryBuffer;
			void* outputKeyBuffer = (void*)( (uint8_t*)temporaryBuffer + def.pSumBuffer );
			void* outputValueBuffer = (void*)( (uint8_t*)temporaryBuffer + def.pSumBuffer + def.keyOutBuffer );

			uint32_t numberOfBlocks = div_round_up( numberOfInputs, m_config.radixSortBlockSize );

			for( int i = 0; ( startBits + i * 8 ) < endBits; i++ )
			{
				uint32_t bitLocation = startBits + i * 8;

				// counter
				{
					ShaderArgument args;
					args.add( inputKeyBuffer );
					args.add( numberOfInputs );
					args.add( temporaryBuffer );
					args.add( bitLocation );
					m_shader->launch( "blockCount", args, numberOfBlocks, 1, 1, 32, 1, 1, stream );
				}
				// Prefix Sum
				{
					// OroStopwatch oroStream( stream );
					// oroStream.start();

					ShaderArgument args;
					args.add( temporaryBuffer );
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
						args.add( temporaryBuffer );
						args.add( bitLocation );
						m_shader->launch( "reorderKeyPair", args, numberOfBlocks, 1, 1, 32, 1, 1, stream );
					}
					else
					{
						ShaderArgument args;
						args.add( inputKeyBuffer );
						args.add( outputKeyBuffer );
						args.add( numberOfInputs );
						args.add( temporaryBuffer );
						args.add( bitLocation );
						m_shader->launch( "reorderKey", args, numberOfBlocks, 1, 1, 32, 1, 1, stream );
					}

					// oroStream.stop();
					// float ms = oroStream.getMs();
					// oroStreamSynchronize( stream );

					// printf( "reorder %f ms\n", ms );
				}
				std::swap( inputKeyBuffer, outputKeyBuffer );
				std::swap( inputValueBuffer, outputValueBuffer );
			}

			// todo odd case.
		}
    private:
        std::unique_ptr<Shader> m_shader;
        Config m_config;
    };
}