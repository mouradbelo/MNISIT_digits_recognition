// T.Grenier
// Testing functions of common_functions.h
// g++ test_common_functions.cpp common_functions.cpp -o test_common_functions -ffast-math -fopenmp -O3
#include <iostream>
#include <string>
#include <ctime>
#include <ratio>
#include <chrono>
#include <cmath>

#define FLOAT_TYPE_TYPE float
#include "common_functions.h"

using namespace std;

void print(const FLOAT_TYPE *v, unsigned int n, string txt)
{
 cout << txt << "[" << n <<"]= { ";
 for(unsigned int i=0; i<n-1; ++i)
	 cout << v[i] <<", ";
 cout << v[n-1] << " }" << endl;
}

int main()
{
// Test compialtion options
#if defined(USE_OMP_COMMON_FUNCTIONS) && defined(_OPENMP)
    cout << " OPENMP is activated : great! " << endl;
#else
    cout << " OPENMP is not activated (good for debug)" << endl;
#endif

#ifdef __FAST_MATH__
    cout << " fast-math is activated : great! " << endl;
#else
    cout << " fast-math is strangly not activated " << endl;
#endif

// Test Sigmoid functions
/*
InitSigmoid();
cout << " ** test sigmoid g_( 0 ) = " << g_( 0.0 ) << endl;
cout << " ** test sigmoid g_( 1 ) = " << g_( 1.0 ) << endl;
cout << " ** test sigmoid g_(-1 ) = " << g_(-1.0 ) << endl;
cout << " ** test sigmoid g_( 6 ) = " << g_( 6.0 ) << endl;
cout << " ** test sigmoid g_(-6 ) = " << g_(-6.0 ) << endl;
cout << " ** test sigmoid g_( 9 ) = " << g_( 9.0 ) << endl;
cout << " ** test sigmoid g_(-9 ) = " << g_(-9.0 ) << endl;
*/

cout << " ** test sigmoid g( 0 ) = " << g( 0.0 ) << endl;
cout << " ** test sigmoid g( 1 ) = " << g( 1.0 ) << endl;
cout << " ** test sigmoid g(-1 ) = " << g(-1.0 ) << endl;
cout << " ** test sigmoid g( 6 ) = " << g( 6.0 ) << endl;
cout << " ** test sigmoid g(-6 ) = " << g(-6.0 ) << endl;
cout << " ** test sigmoid g( 9 ) = " << g( 9.0 ) << endl;
cout << " ** test sigmoid g(-9 ) = " << g(-9.0 ) << endl;


// Test vector manipulations
unsigned int n = 10;

    FLOAT_TYPE *v1=nullptr; 	allocate( &v1, n);
    FLOAT_TYPE *v2=nullptr;  allocate( &v2, n);
    FLOAT_TYPE *v3=nullptr;  allocate( &v3, n);

    zeros(v1, n);
    print(v1, n, "zeros v1");

    ones(v1, n);
    print(v1, n, "ones v1");

	for( unsigned int i=0; i<n; ++i)
	{
		v1[i] = i;
		v2[i] = i*i;
	}
    print(v1, n, "v1");
    print(v2, n, "v2");
	
	cout << " ** squared norm v1 = " << norm_v_sqr( v1 ,n )     << endl; // 285
	cout << " **          v1'.v1 = " << dot_product( v1, v1, n) << endl; // 285 
	cout << " ** || v1 - v2||^2 = "  << norm_2v_sqr( v1, v2, n ) << endl;
	


	sum_2v(v3, v1, v2, n);    // { 0, 2, 6, 12, 20, 30, 42, 56, 72, 90 }
    print(v3, n, " ** v3 = v1+v2");
	
	sub_2v(v3, v1, v2, n);    // { 0, 0, -2, -6, -12, -20, -30, -42, -56, -72 }
    print(v3, n, " ** v1-v2");
	
	mul_v_s(v3, v1, 2, n);    //  { 0, 2, 4, 6, 8, 10, 12, 14, 16, 18 }
    print(v3, n, " ** 2.v1");
	
	cout << " ** v1'.v2 = " << dot_product( v1, v2, n) << endl; // 2025

    sum_2v(v1, v1, v2, n);    // { 0, 2, 6, 12, 20, 30, 42, 56, 72, 90 }
    print(v3, n, " ** v1 += v2 : ");


    destroy(&v1);
    destroy(&v2);
    destroy(&v3);
	cout << "end." << endl;

return 0;
}
