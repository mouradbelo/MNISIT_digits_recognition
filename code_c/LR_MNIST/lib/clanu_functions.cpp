// T. Grenier - Clanu 2017-2018 - INSA Lyon - GE
// g++ mnist_train_lrgd.cpp common_functions.cpp -o mnist_train_lrgd -ffast-math -fopenmp -O3

#include <iostream>
#include <string>
#include <vector>
#include <algorithm>
#include <fstream>
#include <iterator>
#ifdef _OPENMP
#include <omp.h>
#endif

#include "clanu_functions.h"
#include "common_functions.h"

using namespace std;

///////////////////////////////////////////////////////////
//           START YOUR MODIFICATIONS HERE               //
///////////////////////////////////////////////////////////
 float Accuracy ( float** theta, float** testX, float* testY, unsigned int n,unsigned int m)
 {
int count=0;
     /*float* res=nullptr;
     allocate (&res,m); // stocke les m chiffres prédis à comparés avec les vrais chiffres dans y
     zeros(res,m);*/
     for (unsigned int i=0;i<m;i++)// m imagette de la base de test
     {

         FLOAT_TYPE *prob=nullptr; allocate(&prob, 10);
         float max_prob;
         unsigned int c_prob = 0;
     prob[0] = g( dot_product( theta[0], testX[i], n ) );
     max_prob = prob[0];
     for(unsigned int c=1; c<10; c++)
         {
         prob[c] = g( dot_product( theta[c], testX[i], n ) );
         if( max_prob < prob[c])
             {
             max_prob = prob[c];
             c_prob   = c;
             }
         }
count+=(c_prob==testY[i])?1:0;
     /*if (c_prob==testY[i])
         count++;*/
     //res[i]=c_prob;
     }

     /*for (unsigned int i=0;i<m;i++)
         if (res[i]==testY[i])
             count++;*/

     return (100)*float(count)/m;
}
 void readTheta(string fileName, float** theta, int row, int col)
 {
     ifstream file(fileName, ios::in);// ouvrir le fichier en lecture
     if (file)
     {
        for (int i(0);i<row;i++)
            for (int j(0);j<col;j++)
            {
                file >> theta[i][j]; // on lit jusqu'à l'espace ou le saut de ligne
            }
        file.close();
     }
     else
        cerr <<"Erreur, impossible d'ouvir le fichier"<<endl; // on affiche un message d'erreur à la sortie d'erreur si le fichier n'a pu être ouvert
 }
void writeTheta(string fileName, float ** theta, int row, int col)
{
    ofstream file(fileName, ios::out | ios::trunc); // on ouvre le fichier en écriture
    if (file)
    {
        for (int i(0);i<row;i++)
        {
            for (int j(0);j<col;j++)
             {
                 file<< theta[i][j]<<" "; // on écrit la valeur de la colonne séparée d'espaces
             }
            file<< endl;// on fait un saut de ligne pour écrire la nouvelle ligne de theta
        }
        file.close();
     }
     else
         cerr <<"Erreur, impossible d'ouvir le fichier"<<endl;
}


























///////////////////////////////////////////////////////////
//           DO NOT MODIFY AFTER THIS LINE               //
///////////////////////////////////////////////////////////

void print(const FLOAT_TYPE *v, unsigned int n, string txt)
{
 cout << txt << "[" << n <<"]= { ";
 for(unsigned int i=0; i<n-1; ++i)
	 cout << v[i] <<", ";
 cout << v[n-1] << " }" << endl;
}

// Read the file \param filename and complete the matrix \param M of \param nb_lines by \param nb_cols elements with the content of the csv file. 
/*  The matrix \param M will be allocated by this function.
		\param filename : name of the csv file
		\param M : output matrix with content of the file
		\param nb_lines : number of lines reads 
		\param nb_cols  : number of entry per line (features)
*/
void loadCSV_to_matrix( char *filename, FLOAT_TYPE ***M,  unsigned int *nb_lines, unsigned int  *nb_cols)
{
    std::string delimiter = ","; // set the character used as separator
    ifstream file(filename);     // set filename to the stream
    if (file.is_open() != true ) // try to open the file
        {						 // if trouble return with error message
        cerr << " ** can't open file : " << filename << endl;
        return;
        }

    string line;				//else : start file the analysis
    std::vector< string > lines;  
	while(getline(file,line))   // First, Read all lines
		lines.push_back(line);  // and store them in "lines"
	file.close();				// close the file
	
	// Get size of data
    *nb_lines = lines.size();	// Second, determine the number of lines
    *nb_cols = 1 + std::count( lines[0].begin(), lines[0].end(), ','); // idem with the number of features

    // creating matrix M and filling it with csv values
    allocate( M, *nb_lines, *nb_cols );   // and allocate the memory space for M
#if defined(_OPENMP)
    #pragma omp parallel for                            // Concurrency (or parallel) for loop
#endif
    for(unsigned int i=0; i<*nb_lines; ++i) //Third, split each line
        {
        size_t pos = 0;
        std::string token;					
        unsigned int j = 0;
        while ((pos = lines[i].find(delimiter)) != std::string::npos) // search the separator token
            {
            token = lines[i].substr(0, pos); 				// extract the substring stopping at ','
            (*M)[i][j] = std::stof(token);  				// convert string to FLOAT_TYPE
            lines[i].erase(0, pos + delimiter.length()); 	// remove the analyzed number
            j++; 											// go to next entry in M
            }
        }
}

// Extract the features from the matrix \param M to \param X. X is a matrix of size nb_lines by nb_cols-1 (as the first column is the digit, ie the label)
void extract_features_from_CSV(FLOAT_TYPE **X, const FLOAT_TYPE * const* M, unsigned int nb_lines, unsigned int nb_cols)
{
    for(unsigned int i=0; i<nb_lines; ++i)
        for(unsigned int j=1; j<nb_cols; ++j)
            X[i][j-1] = M[i][j];

}

// Extract the labels from the matrix \param M to \param y. y is a vector of size nb_lines and corresponds to the first column of M.
void extract_labels_from_CSV(FLOAT_TYPE *y, const FLOAT_TYPE * const *M, unsigned int nb_lines)
{
     for(unsigned int i=0; i<nb_lines; ++i)
         y[i] = M[i][0];
}




