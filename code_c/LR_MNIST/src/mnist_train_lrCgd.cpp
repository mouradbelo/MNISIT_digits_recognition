// Clanu 2017-2018 - INSA Lyon - GE

#include <iostream>
#if defined(_OPENMP)
    #include <omp.h>
#endif

#define FLOAT_TYPE float

#include "common_functions.h"
#include "clanu_functions.h"
#include "timing_functions.h"



using namespace std;

int main(int argc, char *argv[])
{
    // Test some compialtion options
    #if defined(_OPENMP)
        cout << " OPENMP is activated  : great! " << endl;
    #else
        cout << " OPENMP is not activated  (good for debug)" << endl;
    #endif

    #ifdef __FAST_MATH__
        cout << " fast-math is activated : great! " << endl;
    #else
        cout << " fast-math is strangely not activated " << endl;
    #endif


        ///////////////////////////////////////////////////////////
        //           START YOUR MODIFICATIONS HERE               //
        ///////////////////////////////////////////////////////////
        FLOAT_TYPE tau;
        unsigned int max_it;

        if( argc < 5)
            {
                cerr << " Usage : " << argv[0] << " train.csv test.csv max_it tau" << endl;
                return -1;
            }

        max_it = stoul(argv[3]);
        tau    = stof(argv[4]);
        // Summarizing options
        cout << " ** summarize options : " << endl;
        cout << " \t Training file : " << argv[1] << endl;
        cout << " \t Testing  file : " << argv[2] << endl;
        cout << " \t max_it = " << max_it << endl;
        cout << " \t tau    = " << tau    << endl;



        cout << "Reading and initializing ... This may take a while (~20-30s) " << endl;
        tic();

        // read TRAINING CSV file
        FLOAT_TYPE **CSV=nullptr;
        unsigned int CSV_m, CSV_n;
        loadCSV_to_matrix( argv[1], &CSV,  &CSV_m, &CSV_n);

        // Extract features X and labels y
        unsigned int m = CSV_m;
        unsigned int n = CSV_n - 1; // the first column contains the labels
        FLOAT_TYPE **X = nullptr; allocate( &X, m, n);
        FLOAT_TYPE  *y = nullptr; allocate( &y, m);

        extract_features_from_CSV( X, CSV, CSV_m, CSV_n );
        extract_labels_from_CSV  ( y, CSV, CSV_m );
        destroy( &CSV, CSV_m);

        // Read TESTING CSV file
        loadCSV_to_matrix( argv[2], &CSV,  &CSV_m, &CSV_n);

        // Extract features test_X and labels test_y
        unsigned int test_m = CSV_m;
        FLOAT_TYPE **test_X = nullptr; allocate( &test_X, m, n);
        FLOAT_TYPE  *test_y = nullptr; allocate( &test_y, m);

        extract_features_from_CSV( test_X, CSV, test_m, CSV_n );
        extract_labels_from_CSV  ( test_y, CSV, test_m );
        destroy( &CSV, CSV_m);


        // Allocate Theta variable
        FLOAT_TYPE **Theta=nullptr; allocate(&Theta, 10, n);

        tac();
        float t1=duration();
        cout << "Reading and initialization time : " << duration() << "s " << endl;

        // Training
        FLOAT_TYPE cumulative_error;
        zeros(Theta, 10, n);
        float t=0;
        int iter=0; // itération correspondant à la meilleur précision
        float precision=Accuracy(Theta,test_X,test_y,n,test_m);
        float bestPrecision=precision;// on initialise la précision pour theta=0
        writeTheta(argv[5],Theta,10,n);// on écrit theta correspondant à cette précision

        FLOAT_TYPE beta_k=0;
                 FLOAT_TYPE* res_k=nullptr;
                 FLOAT_TYPE* d_c_k_1=nullptr;
                 FLOAT_TYPE* grad_k=nullptr;
                 FLOAT_TYPE* grad_k_1=nullptr;
                 allocate(&d_c_k_1,n);
                 allocate(&grad_k,n);
                 allocate(&grad_k_1,n);
                 allocate(&res_k,n);

        for(unsigned int k=0; k < max_it; k++)
        {
           /* for (int i=0;i<n;i++)
            {
                grad_k[i]=0;
                grad_k_1[i]=0;
                d_c_k_1[i]=0;
                res_k[i]=0;
            }*/
            if(k!=0)
            {
            cumulative_error = 0;
            tic();
            #if defined(_OPENMP)
            #pragma omp parallel for reduction(+:cumulative_error)  // Concurrency (or parallel) for loop
            #endif
            for(unsigned int c=0; c<10; c++)
            {

                FLOAT_TYPE *theta_c_k = Theta[c];                    //  linked on Theta, for easier reading
                FLOAT_TYPE *d_c_k=nullptr; allocate(&d_c_k,n);       // ( definied and allocated here for concurrency )
                zeros( d_c_k, n);

                   copy1(grad_k_1,grad_k,n);
                for(unsigned int i=0; i<m; i++)
                {

                    FLOAT_TYPE y_c_i = (y[i]==c)?1.0:0.0;            // y_c_i = 1 if y[i] == c
                                                            //         0  otherwise
                    FLOAT_TYPE h_theta_c_i =
                        g( dot_product( theta_c_k, X[i], n ) ) - y_c_i; // h_theta_c_i = g( theta_c_k . X[i] ) - y_c_i
                    mac_v_v_s( grad_k, X[i], h_theta_c_i, n );    //  grad_k += X[i] * h_theta_c_i
                    cumulative_error += abs(h_theta_c_i);
                  }
                mul_v_s(grad_k,grad_k,1.0/m,n);
                    if (k==1)
                    {

                         copy1(d_c_k,grad_k,n);
                         mul_v_s(d_c_k,d_c_k,-1,n);
                          //copy1(grad_k_1,grad_k,n);

                    }
                    else
                    {

                        sub_2v(d_c_k_1,grad_k,grad_k_1,n);
                        beta_k=dot_product(d_c_k_1,grad_k,n);
                        beta_k/=(norm_v_sqr(grad_k_1,n));

                        mul_v_s(d_c_k,d_c_k_1,beta_k,n);
                        mul_v_s(grad_k,grad_k,-1,n);
                        copy1(d_c_k_1,d_c_k,n);
                       // copy1(grad_k_1,grad_k,n);
                        sum_2v(d_c_k,grad_k,res_k,n);
                        //*d_c_k=-(*grad_k)+(beta_k)*(*d_c_k_1);
                    }
                                // ( used for evolution tracking )

                mul_v_s( d_c_k, d_c_k, tau , n);        //  d_c_k *=  tau
                sum_2v( theta_c_k, theta_c_k, d_c_k, n);        //  theta_c_k+1 = theta_c_k + d_c_k
                destroy(&d_c_k);

                /*destroy(&grad_k );
                destroy(&res_k);*/
            }

            tac();
            cout << "it : " << k << "\t time : " << duration() << " s\t error : "<< cumulative_error/(10*m);
            t+=duration();
            precision=Accuracy(Theta,test_X,test_y,n,test_m);
            if (precision>bestPrecision)
            {
                writeTheta(argv[5],Theta,10,n); // on sauvegarde theta pour la meilleure précision
                bestPrecision=precision;
                iter=k;
            }
            cout <<endl;
            if (k%5==0)
                cout << "Precision de : "<<precision;
            cout << endl;
        }
            }
        //readTheta("/home/mbelhamiss/Bureau/Clanu/bestTheta.csv",Theta,10,n);
        //cout<<Accuracy(Theta,test_X,test_y,n,test_m)<<endl;
        /*for (int i(0);i<10;i++)
        {for (int j(0);j<n;j++)
                cout << Theta[i][j]<<" ";
        cout<< endl;}*/

        cout << "meilleure précision de "<<bestPrecision <<" pour "<<iter <<" itérations."<<endl;



        // Test with data at "test_index" from test.csv
        unsigned int test_index=18;
        FLOAT_TYPE *prob=nullptr; allocate(&prob, 10);
        FLOAT_TYPE max_prob;
        unsigned int c_prob = 0;
        prob[0] = g( dot_product( Theta[0], test_X[test_index], n ) );
        max_prob = prob[0];
        for(unsigned int c=0; c<10; c++)
            {
            prob[c] = g( dot_product( Theta[c], test_X[test_index], n ) );
            if( max_prob < prob[c])
                {
                max_prob = prob[c];
                c_prob   = c;
                }
            }
        cout << " The value at " << test_index << " should be : " << test_y[test_index] << " and the prediction done give : "  << c_prob;
        if( test_y[test_index] == c_prob ) cout << "  Good prediction :) !!" << endl;
        else cout << " Prediction error :( !!" << endl;
        print(prob, 10, " probabilities : ");






        // free memory
        destroy( &prob   );
        destroy( &y      );
        destroy( &test_y );

        destroy( &Theta, 10);
        destroy( &X, m     );
        destroy( &test_X, test_m );

        cout << " end." << endl;
        cout << "Temps total: "<<t+t1;
        return 0;


cout << " end." << endl;
return 0;
}
