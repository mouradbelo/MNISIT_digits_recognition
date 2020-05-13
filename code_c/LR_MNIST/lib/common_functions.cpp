// T. Grenier - Clanu 2017-2018 - INSA Lyon - GE
//compile with -ffast-math -fopenmp 

///////////////////////////////////////////////////////////
//           DO NOT MODIFY THIS FILE                     //
///////////////////////////////////////////////////////////

#include "common_functions.h"

#include <cstring>
#include <algorithm>
#include <cmath>

#ifdef _OPENMP
    #include <omp.h>
#endif

using namespace std;


void InitSigmoid()
{
    FLOAT_TYPE x;
    for( unsigned int i=0; i<Sigmoid_Resolution; ++i)
        {
        x = fma( Sigmoid_Alpha , i, Sigmoid_Beta); // x = Sigmoid_Alpha * i + Sigmoid_Beta
        Sigmoid[i] = 1.0 / ( exp(-x) + 1);
        }
}

FLOAT_TYPE g_(FLOAT_TYPE s)
{
    if( s > Sigmoid_MaxBound ) return FLOAT_TYPE(1);
    if( s < Sigmoid_MinBound ) return FLOAT_TYPE(0);

    return Sigmoid[ lrint ( fma( s , Sigmoid_inv_Alpha, Sigmoid_inv_Beta) ) ];
}


#if defined(_OPENMP)
    #pragma omp declare simd
#endif
inline FLOAT_TYPE distsq(FLOAT_TYPE x, FLOAT_TYPE y)
{
    return (x-y) * (x-y);
}


// Compute the squared norm of one vector r = ||v||²
FLOAT_TYPE norm_v_sqr( FLOAT_TYPE *v, unsigned int n)
{
    FLOAT_TYPE norm ( 0.0 );
    for (unsigned int i=0; i<n ; i++)
            norm += v[i] * v[i] ;
    return norm;
}


// Compute the squared norm of two vectors  (squared euclidian distance): r = || v1 - v2 ||²
FLOAT_TYPE norm_2v_sqr( const FLOAT_TYPE *v1, const FLOAT_TYPE *v2, unsigned int n)
{
    FLOAT_TYPE norm ( 0.0 );
#if defined(_OPENMP)
    #pragma omp
#endif
    for (unsigned int i=0; i<n ; i++)
            norm += distsq( v1[i], v2[i]) ;
    return norm;
}


// r = v1 + v2
void sum_2v(FLOAT_TYPE *r, const FLOAT_TYPE *v1, const FLOAT_TYPE*v2, unsigned int n)
{
   if( r != v1) std::memcpy(r, v1, sizeof(FLOAT_TYPE) * n);
    for (unsigned int i=0; i<n ; i++)
        r[i] += v2[i];
}


// r = v1 - v2
void sub_2v(FLOAT_TYPE *r, const FLOAT_TYPE *v1, const FLOAT_TYPE*v2, unsigned int n)
{
   if( r != v1)  std::memcpy(r, v1, sizeof(FLOAT_TYPE) * n);
    for (unsigned int i=0; i<n ; i++)
        r[i] -= v2[i];
}


// r = v * s
void mul_v_s(FLOAT_TYPE *r, const FLOAT_TYPE *v1, FLOAT_TYPE s, unsigned int n)
{
    if( r != v1) std::memcpy(r, v1, sizeof(FLOAT_TYPE) * n);
    for (unsigned int i=0; i<n ; i++)
        r[i] *= s;
}

// r += v * s
void mac_v_v_s(FLOAT_TYPE *r, const FLOAT_TYPE *v, FLOAT_TYPE s, unsigned int n)
{
    for (unsigned int i=0; i<n ; i++)
        r[i] +=  v[i] * s;
}

// r = v1 .* v2
void element_product( FLOAT_TYPE *r, const FLOAT_TYPE *v1, const FLOAT_TYPE *v2, unsigned int n)
{
   if( r != v1) std::memcpy(r, v1, sizeof(FLOAT_TYPE) * n);
    for (unsigned int i=0; i<n ; i++)
        r[i] *= v2[i];
}


// s = v1' * v2 
FLOAT_TYPE dot_product( const FLOAT_TYPE *v1, const FLOAT_TYPE *v2, unsigned int n)
{
    FLOAT_TYPE norm ( 0.0 );
    for (unsigned int i=0; i<n ; i++)
            norm += v1[i] * v2[i];
    return norm;
}

// allocate memory for v, a vector of n floats
bool allocate(FLOAT_TYPE **v, unsigned int n)
{
	if( *v == nullptr )
        *v = new FLOAT_TYPE[n];
	return (*v)?true:false;
}

// free space
void destroy( FLOAT_TYPE **v)
{
	if( *v != nullptr )
		 delete[] *v;
	*v = nullptr;
}

// allocate memory for M, a matrix of m by n floats
bool allocate(FLOAT_TYPE ***M, unsigned int m, unsigned int n)
{
	if( *M != nullptr ) return false;
    *M = new FLOAT_TYPE*[m];
#if defined(_OPENMP)
    #pragma omp parallel for
#endif
    for (unsigned int i = 0; i < m; i++)
        {
        (*M)[i] = new FLOAT_TYPE[n];
        }
	return true;	
}

// free space of matrix M
bool destroy(FLOAT_TYPE ***M, unsigned int m)
{
	if( *M != nullptr )
		{
#if defined(_OPENMP)
    #pragma omp parallel for
#endif
        for (unsigned int i = 0; i < m; i++)
            delete[] (*M)[i];
		delete[] *M;
		*M = nullptr;
        return true;
		}
    return false;
}

// copy content of v to u. Memory spaces must be reserved first
void copy1(FLOAT_TYPE *u, const FLOAT_TYPE *v, unsigned int n)
{
   if( u != v)  std::memcpy(u, v, sizeof(FLOAT_TYPE) * n);
}

// set to 0 the m by n elements of M
void zeros( FLOAT_TYPE **M, unsigned int m, unsigned int n )
{
    for(unsigned int i=0; i<m; ++i)
        for(unsigned int j=0; j<n; ++j)
            M[i][j] = 0.0;
}

// set to 0 the n elements of v
void zeros( FLOAT_TYPE *v, unsigned int n )
{
    for(unsigned int i=0; i<n; ++i)
        v[i] = 0.0;
}


// set to 1 the m by n elements of M
void ones(FLOAT_TYPE **M, unsigned int m, unsigned int n )
{
    for(unsigned int i=0; i<m; ++i)
        for(unsigned int j=0; j<n; ++j)
            M[i][j] = 1.0;
}

// set to 1 the n elements of v
void ones(FLOAT_TYPE *v, unsigned int n )
{
    for(unsigned int i=0; i<n; ++i)
        v[i] = 1.0;
}
