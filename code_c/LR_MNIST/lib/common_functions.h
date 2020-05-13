// T. Grenier - Clanu 2017-2018 - INSA Lyon - GE

///////////////////////////////////////////////////////////
//           DO NOT MODIFY THIS FILE                     //
///////////////////////////////////////////////////////////

#ifndef _common_functions_h_
#define _common_functions_h_


#define FLOAT_TYPE float

#include <algorithm>
#include <cmath>


#define Sigmoid_Resolution 10001

static const FLOAT_TYPE Sigmoid_MinBound =  -10.0;
static const FLOAT_TYPE Sigmoid_MaxBound =   10.0;
//! this is the precomputed values used for Sigmoid [-6:+6]
static const FLOAT_TYPE Sigmoid_Alpha = 1.0 * (Sigmoid_MaxBound - Sigmoid_MinBound) / (Sigmoid_Resolution - 1);
static const FLOAT_TYPE Sigmoid_Beta  = Sigmoid_MinBound;

static const FLOAT_TYPE Sigmoid_inv_Alpha = 1.0 / Sigmoid_Alpha;
static const FLOAT_TYPE Sigmoid_inv_Beta  = -Sigmoid_Beta / Sigmoid_Alpha;

static FLOAT_TYPE Sigmoid[Sigmoid_Resolution];

//! Compute value for sigmoid array
void InitSigmoid();

//! Approximated implementation of Sigmoid
FLOAT_TYPE g_(FLOAT_TYPE s);



//! Inline implementation of Sigmoid
inline FLOAT_TYPE g(FLOAT_TYPE s)
{
    return 1.0/(1.0+exp(-s) );
}

//! compute square distance of scalar \param x and \param y
inline FLOAT_TYPE distsq(FLOAT_TYPE x, FLOAT_TYPE y);

//! Compute the squared norm of one vector $r = ||v||^2$
FLOAT_TYPE norm_v_sqr( FLOAT_TYPE *v, unsigned int n);

//! Compute the squared norm of two vectors  (squared euclidian distance): $r = || v1 - v2 ||^2$
FLOAT_TYPE norm_2v_sqr( const FLOAT_TYPE *v1, const FLOAT_TYPE *v2, unsigned int n);

//! $r = v1 + v2$. All vectors involved must be allocated before calling this function.
void sum_2v(FLOAT_TYPE *r, const FLOAT_TYPE *v1, const FLOAT_TYPE *v2, unsigned int n);

//! $r = v1 - v2$. All vectors involved must be allocated before calling this function.
void sub_2v(FLOAT_TYPE *r, const FLOAT_TYPE *v1, const FLOAT_TYPE *v2, unsigned int n);

//! $r = v * s$. All vectors involved must be allocated before calling this function.
void mul_v_s(FLOAT_TYPE *r, const FLOAT_TYPE *v1, FLOAT_TYPE s, unsigned int n);

//! $r += v * s$. All vectors involved must be allocated before calling this function.
void mac_v_v_s(FLOAT_TYPE *r, const FLOAT_TYPE *v, FLOAT_TYPE s, unsigned int n);

//! $s = v1' * v2$. All vectors involved must be allocated before calling this function. 
FLOAT_TYPE dot_product( const FLOAT_TYPE *v1, const FLOAT_TYPE *v2, unsigned int n);

//! $r = v1 .* v2$. Elements per element product. All vectors involved must be allocated before calling this function.
void element_product( FLOAT_TYPE *r, const FLOAT_TYPE *v1, const FLOAT_TYPE *v2, unsigned int n);

//! Allocate memory for \param n FLOAT_TYPE elements in \param v. \param v will be allocated only if it is previously set to nullptr.
//! Example of usage : FLOAT_TYPE *v=nullptr;  allocate( &v , 1000 );
//! \param v must be desallocate with destroy : destroy_v( &v );
bool allocate( FLOAT_TYPE **v, unsigned int n);

//! Free memory for vector \param v of FLOAT_TYPEs. \param v is set to nullptr.
void destroy( FLOAT_TYPE **v);

//! Allocate memory for matrix \param M of \param m by \param n FLOAT_TYPE elements. \param M will be allocated only if it is previously set to nullptr.
bool allocate( FLOAT_TYPE ***M, unsigned int m, unsigned int n);

//! Free memory for matrix \param M of \param m elements of type FLOAT_TYPE *. After deallocating memory \param M will set to nullptr by this function.
bool destroy( FLOAT_TYPE ***M, unsigned int m);

//! Copy \param n FLOAT_TYPE elements of \param v to \param u
void copy1(FLOAT_TYPE *u, const FLOAT_TYPE *v, unsigned int n);

//! Set all \param m x \param n elements of \param M to 0
void zeros( FLOAT_TYPE **M, unsigned int m, unsigned int n );

//! Set all \param n elements of \param v to 0
void zeros( FLOAT_TYPE *v, unsigned int n );

//! Set all \param m x \param n elements of \param M to 1
void ones( FLOAT_TYPE **M, unsigned int m, unsigned int n );

//! Set all \param n elements of \param v to 1
void ones( FLOAT_TYPE *v, unsigned int n );

#endif // _common_functions_h
