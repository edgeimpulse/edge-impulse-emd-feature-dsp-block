#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <gsl/gsl_math.h>
#include <armadillo>
#include <filesystem>
#include <experimental/filesystem>

#include "eemd.h"

const double pi = M_PI;
const size_t ensemble_size = 500;
const unsigned int S_number = 4;
const unsigned int num_siftings = 50;
const double noise_strength = 0.02;
const unsigned long int rng_seed = 0;
const char outfile[] = "trial.out";
const size_t N = 1500; // Window size is equal to 15 ms

// This software is composed of 3 parts
// The first part is the data loader which loads all the dataset into a one
// giant std vector.
// Second part is the EMD computation part which contains the C
// code that is used to compute the IMF
// The third part statistical part which contains the computation
// of the mean, std dev, skewness, etc...
// Finally the fourth part, saving all of these extracted features to a
// file locally. 


// This file is only necessary to parse the original raw datasets
// and load all of them into one matrix and then save the matrix to 
// We have 11 classes, which means 11 files in totals as a results.
void load_all_datasets(std::vector<arma::mat>& datasets, int class_num)
{
  arma::mat file;
  for (size_t i=1; i < 9; ++i)
  {
    bool state =
    file.load("../../datasets/raw_datasets/Sensorless_Drive_Diagnosis_Data_Set/class"
    + std::to_string(class_num) + "/class" + std::to_string(class_num) + "_Parameterset" 
    + std::to_string(i) + ".txt");
    std::cout << "Print the state of the loading:.. "<< state << std::endl;
    if (state == false)
    {
      //see later what to do
    }
    datasets.push_back(file);
  }
}

int main(void)
{
  std::vector<arma::mat> dataset_class1;
  load_all_datasets(dataset_class1, 1);

  
  // libeemd_error_code err;
 
  // // We have only two phases as input signals from the motors.
  // double* phase_1_input_vector = (double*) malloc(N*sizeof(double));
  // double* phase_2_input_vector = (double*) malloc(N*sizeof(double));
  // memset(phase_1_input_vector, 0x00, N*sizeof(double));
  // memset(phase_2_input_vector, 0x00, N*sizeof(double));

  // size_t M = emd_num_imfs(N);
  // double* output_phase_1 = (double*)malloc(M*N*sizeof(double));
  // double* output_phase_2 = (double*)malloc(M*N*sizeof(double));

  // arma::mat phase1;


  // // Decut the signals into 15 ms, which means 1500 lines of code.
  // for (size_t i=0; i < N; ++i)
  // {
  //   input_vector[i]= phase1(i, 1);
  // }

  // // Run CEEMDAN
  // err = ceemdan(input_vector, N, outp, M, ensemble_size, noise_strength, S_number, num_siftings, rng_seed);
  // if (err != EMD_SUCCESS)
  // {
  //   emd_report_if_error(err);
  //   exit(1);
  // }

  // // Write output to file
  // // First write the signals it self as the first line of the file
  // // Second write the IMFs from the output variable to the same file
  // // Finally the file contains the original signals and the IMFS and the res.
  // FILE* fp = fopen(outfile, "w");
  // for (size_t j=0; j<N; j++)
  // {
  //   fprintf(fp, "%f ", input_vector[j]);
  // }
  // fprintf(fp, "\n");
  // for (size_t i=0; i<M; i++)
  // {
  //   for (size_t j=0; j<N; j++)
  //   {
  //     fprintf(fp, "%f ", outp[i*N+j]);
  //   }
  //   fprintf(fp, "\n");
  // }
  // printf("Done!\n");
  // // Cleanup
  // fclose(fp);
  // free(input_vector); input_vector = NULL;
  // free(outp); outp = NULL;
}
