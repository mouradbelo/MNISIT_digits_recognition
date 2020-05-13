// T. Grenier - Clanu 2017-2018 - INSA Lyon - GE

///////////////////////////////////////////////////////////
//           DO NOT MODIFY THIS FILE                     //
///////////////////////////////////////////////////////////

#ifndef _timing_functions_h
#define _timing_functions_h


#include <ctime>
#include <ratio>
#include <chrono>


std::chrono::steady_clock::time_point tic_time;
std::chrono::steady_clock::time_point tac_time;

//! Start the chronometer.
void tic()
{
tic_time = std::chrono::steady_clock::now();
}

//! Stop the chronometer.
void tac()
{
tac_time = std::chrono::steady_clock::now();
}

//! Return time in seconds elapsed between last tic() and tac() calls.
double duration()
{
    std::chrono::duration<double> time_span = std::chrono::duration_cast< std::chrono::duration<double> >(tac_time - tic_time);
    return time_span.count();
}
#endif 
