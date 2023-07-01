#pragma once

int32_t __float_as_int( float x )
{
	return *(int32_t*)&x;
}
uint32_t __float_as_uint( float x )
{
	return *(uint32_t*)&x;
}
int64_t __double_as_longlong( double x )
{
	return *(int64_t*)&x;
}
inline uint32_t getKeyBits( uint32_t x )
{
	return x;
}
inline uint64_t getKeyBits( uint64_t x )
{
	return x;
}
inline uint32_t getKeyBits( float x )
{
	if( x == 0.0f )
		x = 0.0f;

	uint32_t flip = uint32_t( __float_as_int( x ) >> 31 ) | 0x80000000;
	return __float_as_uint( x ) ^ flip;
}
inline uint64_t getKeyBits( double x )
{
	if( x == 0.0 )
		x = 0.0;

	uint64_t flip = uint64_t( __double_as_longlong( x ) >> 63 ) | 0x8000000000000000llu;
	return (uint64_t)__double_as_longlong( x ) ^ flip;
}
