// Clanu 2017-2018 - INSA Lyon - GE

#include <iostream>
#include <cmath>
#define FLOAT_TYPE float

#include "common_functions.h"
#include "clanu_functions.h"
#include "timing_functions.h"

using namespace std;

int main(int argc, char *argv[])
{
if( argc < 3)
    {
        cerr << " Usage : " << argv[0] << "Theta_filename Test_filename.csv" << endl;
        return -1;
    }

// Summarizing options
cout << " ** summarize options : " << endl;
cout << " \t Theta file    : " << argv[1] << endl;
cout << " \t Testing  file : " << argv[2] << endl;

///////////////////////////////////////////////////////////
//           START YOUR MODIFICATIONS HERE               //
///////////////////////////////////////////////////////////
cout << "Reading and initializing ... This may take a while (~20-30s) " << endl;
tic();

// read CSV file
FLOAT_TYPE **CSV=nullptr;
unsigned int CSV_m, CSV_n;
loadCSV_to_matrix( argv[2], &CSV,  &CSV_m, &CSV_n);

// Extract features X and labels y
unsigned int m = CSV_m;
unsigned int n = CSV_n - 1; // the first column contains the labels
FLOAT_TYPE **X = nullptr; allocate( &X, m, n);
FLOAT_TYPE  *y = nullptr; allocate( &y, m);

extract_features_from_CSV( X, CSV, CSV_m, CSV_n );
extract_labels_from_CSV  ( y, CSV, CSV_m );
destroy( &CSV, CSV_m);

FLOAT_TYPE **theta=nullptr; allocate(&theta, 10, n);
readTheta(argv[1],theta,10,n);

tac();
cout << "Reading and initialization time : " << duration() << "s " << endl;

float accuracyChiffre[10]={0,0,0,0,0,0,0,0,0,0};// Chaque élément est le nombre de prédiction correcte pour le chiffre (qui est égale à l'indice)
float nombreChiffre[10]={0,0,0,0,0,0,0,0,0,0};// le nombre totale des chiffres de 0 à 10 dans y

float* res=nullptr;
allocate (&res,m); // stocke les m chiffres prédis à comparés avec les vrais chiffres dans y
zeros(res,m);
for (unsigned int i=0;i<m;i++)// m imagette de la base de test
{
    FLOAT_TYPE *prob=nullptr; allocate(&prob, 10);
    float max_prob;
    unsigned int c_prob = 0;
prob[0] = g( dot_product( theta[0], X[i], n ) );
max_prob = prob[0];
for(unsigned int c=1; c<10; c++)
    {
    prob[c] = g( dot_product( theta[c], X[i], n ) );
    if( max_prob < prob[c])
        {
        max_prob = prob[c];
        c_prob   = c;
        }
    }
nombreChiffre[int(y[i])]++;
res[i]=c_prob;
cout << "Le chiffre prédis pour la ligne "<<i <<" est: "<<res[i]<<endl;
cout << "La bonne valeur est: "<<y[i]<<endl;
if (res[i]==y[i])
{
    cout <<"La prédiction est bonne :) "<<endl;
    accuracyChiffre[int(y[i])]++;
}
else
    cout <<" Mauvaise prédiction :( "<<endl;
}
float precisionTotale=0;
for (int i =0;i<10;i++)
{
    cout<<"La précision du chiffre "<< i <<" est "<<100*accuracyChiffre[i]/nombreChiffre[i]<<"%"<<endl;
    precisionTotale+=   100*accuracyChiffre[i]/nombreChiffre[i];
}
    cout<<"La précision totale est: "<<precisionTotale/10<<endl;
cout << " end." << endl;
return 0;
}
