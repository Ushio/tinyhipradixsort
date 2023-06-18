#include <inttypes.h>
#include <algorithm>
#include <vector>
#include <string>
#include <map>
#include <Orochi/Orochi.h>

#include <intrin.h>
#define SH_ASSERT(ExpectTrue) if((ExpectTrue) == 0) { __debugbreak(); }

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

enum class CompileMode
{
	Release,
	RelwithDebInfo
};
inline void loadAsVector( std::vector<char>* buffer, const char* fllePath )
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
	Shader( const char* filename, const char* kernelLabel, const std::vector<std::string>& includeDirs, const std::vector<std::string>& extraArgs, CompileMode compileMode, bool isNvidia )
	{
		std::vector<char> src;
		loadAsVector( &src, filename );
		SH_ASSERT( 0 < src.size() );
		src.push_back( '\0' );

		orortcProgram program = 0;
		orortcCreateProgram( &program, src.data(), kernelLabel, 0, 0, 0 );
		std::vector<std::string> options;
		options.push_back( "-std=c++11");

		if( isNvidia )
		{
			options.push_back( "--gpu-architecture=compute_70" );
			options.push_back( "-DPLATFORM_NVIDIA=1" );
		}
		for( int i = 0; i < includeDirs.size(); ++i )
		{
			// A space between -I and path should not exist. 
			// This is a workaround for AMD rtc API on hiprtc0503.dll
			options.push_back( "-I" + includeDirs[i] );
		}

		if( compileMode == CompileMode::RelwithDebInfo )
		{
			if( isNvidia )
			{
				options.push_back( "--generate-line-info" );
			}
			else
			{
                options.push_back( "-g" );
			}
		}

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
		SH_ASSERT( compileResult == ORORTC_SUCCESS );

		size_t codeSize = 0;
		orortcGetCodeSize( program, &codeSize );

		std::vector<char> codec( codeSize );
		orortcGetCode( program, codec.data() );

		orortcDestroyProgram( &program );

		orortcResult re;
		oroError e = oroModuleLoadData( &m_module, codec.data() );
		SH_ASSERT( e == oroSuccess );
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
			SH_ASSERT( e == oroSuccess );
			m_functions[name] = f;
		}

		auto params = arguments.kernelParams();
		oroFunction f = m_functions[name];
		oroError e = oroModuleLaunchKernel( f, gridDimX, gridDimY, gridDimZ, blockDimX, blockDimY, blockDimZ, 0, hStream, params.data(), 0 );
		SH_ASSERT( e == oroSuccess );
	}

private:
	oroModule m_module = 0;
	std::map<std::string, oroFunction> m_functions;
};