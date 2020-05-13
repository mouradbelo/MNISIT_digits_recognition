// T.Grenier
// Testing functions of common_functions.h
// g++ test_common_functions.cpp common_functions.cpp -o test_common_functions -ffast-math -fopenmp -O3
#include <iostream>
#include <string>
#include <ctime>
#include <ratio>
#include <chrono>
#include <cmath>
#include "common_functions.h"
#include "timing_functions.h"


using namespace std;


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

InitSigmoid();

//Sigmoid Time efficiency
tic();
    cout << "Computing many non null sigmoid with g_ ... ";
    for(float s=-3.0; s<3.0; s+=0.0001) g_(s);
tac();
cout << "It took me " << duration() << "seconds " << endl;

tic();
    cout << "Computing many non null sigmoid with g  ... ";
    for(float s=-3.0; s<3.0; s+=0.0001) g(s);
tac();
cout << "It took me " << duration() << "seconds " << endl;

cout << "end." << endl;
return 0;
}
