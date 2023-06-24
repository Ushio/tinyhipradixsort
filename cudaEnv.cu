#include <cuda.h>
#include <stdio.h>
#include <vector>
#include <algorithm>
#include <cub/util_allocator.cuh>
#include <cub/device/device_radix_sort.cuh>


#include <intrin.h>
#define SH_ASSERT( ExpectTrue ) \
	if( ( ExpectTrue ) == 0 )   \
	{                           \
		__debugbreak();         \
	}


struct splitmix64
{
	uint64_t x = 0; /* The state can be seeded with any value. */

	uint64_t next()
	{
		uint64_t z = ( x += 0x9e3779b97f4a7c15 );
		z = ( z ^ ( z >> 30 ) ) * 0xbf58476d1ce4e5b9;
		z = ( z ^ ( z >> 27 ) ) * 0x94d049bb133111eb;
		return z ^ ( z >> 31 );
	}
};

class CudaStopwatch
{
public:
	CudaStopwatch( CUstream stream )
	{
		m_stream = stream;
		cudaEventCreateWithFlags( &m_start, cudaEventDefault );
		cudaEventCreateWithFlags( &m_stop, cudaEventDefault );
	}
	~CudaStopwatch()
	{
		cuEventDestroy( m_start );
		cuEventDestroy( m_stop );
	}

	void start() { cuEventRecord( m_start, m_stream ); }
	void stop() { cuEventRecord( m_stop, m_stream ); }

	float getMs()
	{
		cuEventSynchronize( m_stop );
		float ms = 0;
		cuEventElapsedTime( &ms, m_start, m_stop );
		return ms;
	}

public:
	CUstream m_stream;
	CUevent m_start;
	CUevent m_stop;
};

int main()
{
	cuInit( 0 );

	CUdevice cuDevice;
	CUresult res = cuDeviceGet( &cuDevice, 0 );
	if( res != CUDA_SUCCESS )
	{
		printf( "cannot acquire device 0\n" );
		exit( 1 );
	}

	CUcontext cuContext;
	res = cuCtxCreate( &cuContext, 0, cuDevice );
	if( res != CUDA_SUCCESS )
	{
		printf( "cannot create context\n" );
		exit( 1 );
	}

	CUstream stream;
	cuStreamCreate( &stream, CU_STREAM_DEFAULT );

	std::vector<uint32_t> inputs( 160 * 1000 * 1000 );

	uint32_t* inputBuffer;
	uint32_t* outputBuffer;
	cudaMalloc( (void**)&inputBuffer, inputs.size() * sizeof( uint32_t ) );
	cudaMalloc( (void**)&outputBuffer, inputs.size() * sizeof( uint32_t ) );

	void* d_temp_storage = NULL;
	size_t temp_storage_bytes = 0;

	cub::DeviceRadixSort::SortKeys( d_temp_storage, temp_storage_bytes, inputBuffer, outputBuffer, inputs.size(), 0, 32, stream, false );
	cudaMalloc( (void**)&d_temp_storage, temp_storage_bytes );

	splitmix64 rng;
	for (;;)
	{
		for( int i = 0; i < inputs.size(); i++ )
		{
			inputs[i] = rng.next() & 0xFFFFFFFF;
		}
		cuMemcpyHtoD( (CUdeviceptr)inputBuffer, inputs.data(), inputs.size() * sizeof( uint32_t ) );

		CudaStopwatch cudaSw( stream );
		cudaSw.start();

		cudaError_t e = cub::DeviceRadixSort::SortKeys( d_temp_storage, temp_storage_bytes, inputBuffer, outputBuffer, inputs.size(), 0, 32, stream, false );

		cudaSw.stop();
		float ms = cudaSw.getMs();
		cudaStreamSynchronize( stream );

		printf( "%f ms\n", ms );

		break;

		std::vector<uint32_t> outputs( inputs.size() );
		CUresult r = cuMemcpyDtoH( outputs.data(), (CUdeviceptr)outputBuffer, inputs.size() * sizeof( uint32_t ) );
		for( int i = 0; i < outputs.size() - 1; i++ )
		{
			SH_ASSERT( outputs[i] <= outputs[i + 1] );
		}
		std::sort( inputs.begin(), inputs.end() );
		for( int i = 0; i < outputs.size(); i++ )
		{
			SH_ASSERT( outputs[i] == inputs[i] );
		}
	}
}