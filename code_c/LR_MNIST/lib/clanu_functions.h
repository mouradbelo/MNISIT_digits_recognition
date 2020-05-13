// T. Grenier - Clanu 2017-2018 - INSA Lyon - GE
// g++ mnist_train_lrgd.cpp common_functions.cpp -o mnist_train_lrgd -ffast-math -fopenmp -O3
#ifndef _clanu_functions_h
#define _clanu_functions_h

#define FLOAT_TYPE float

#include <string>
#include <cstdio>
#include <cstdlib>
using namespace std;


///////////////////////////////////////////////////////////
//           START YOUR MODIFICATIONS HERE               //
///////////////////////////////////////////////////////////


 float Accuracy (float** theta, float** testX, float* testY, unsigned int n, unsigned int m);
 void readTheta(string fileName, float ** theta, int row, int col);
 void writeTheta(string fileName, float ** theta, int row, int col);












///////////////////////////////////////////////////////////
//           DO NOT MODIFY AFTER THIS LINE               //
///////////////////////////////////////////////////////////

//! Display the content of vector \param v of \param n elements
void print(const FLOAT_TYPE *v, unsigned int n, string txt);

//! Read the file \param filename and complete the matrix \param M of \param nb_lines by \param nb_cols elements with the content of the csv file. 
/*!  The matrix \param M will be allocated by this function.
		\param filename : name of the csv file
		\param M : output matrix with content of the file
		\param nb_lines : number of lines reads 
		\param nb_cols  : number of entry per line (features)
*/
void loadCSV_to_matrix( char *filename, FLOAT_TYPE ***M,  unsigned int *nb_lines, unsigned int  *nb_cols);

//! Extract the features from the matrix \param M to \param X. X is a matrix of size nb_lines by nb_cols-1 (as the first column is the digit, ie the label)
void extract_features_from_CSV(FLOAT_TYPE **X, const FLOAT_TYPE * const* M, unsigned int nb_lines, unsigned int nb_cols);

//! Extract the labels from the matrix \param M to \param y. y is a vector of size nb_lines and corresponds to the first column of M.
void extract_labels_from_CSV(FLOAT_TYPE *y, const FLOAT_TYPE * const *M, unsigned int nb_lines);

#endif // _clanu_functions_h
