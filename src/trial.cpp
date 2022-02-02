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
  // If the user enter the class_num 0 then parse all the dataset entirely with
  // all the classes
  if (class_num == 0)
  {
    for (size_t  j=1; j < 11; ++j)
    {
      for (size_t i=1; i < 9; ++i)
      {
        bool state =
        file.load("../../datasets/raw_datasets/Sensorless_Drive_Diagnosis_Data_Set/class"
        + std::to_string(j) + "/class" + std::to_string(j) + "_Parameterset" 
        + std::to_string(i) + ".txt");
        std::cout << "Loading State: "<< state << " Loading class number: " << j << std::endl;
        if (state == false)
        {
          //see later what to do
        }
        datasets.push_back(file);
      }
    }
  } 
  else // parse class by number the dataset will be related to a specific class
  {
    for (size_t i=1; i < 9; ++i)
    {
      bool state =
      file.load("../../datasets/raw_datasets/Sensorless_Drive_Diagnosis_Data_Set/class"
      + std::to_string(class_num) + "/class" + std::to_string(class_num) + "_Parameterset" 
      + std::to_string(i) + ".txt");
      std::cout << "Loading State: "<< state << "Loading class: "<< class_num << " file num: " << std::endl;
      if (state == false)
      {
        //see later what to do
      }
      datasets.push_back(file);
    }
  }
}


// This function constitutes the main functionality of
// computing the IMF of the signals.
// The signal matrix or vector, should be cut to 1500 and prepared for 
// the dsp. This block does not handle any kind of data preparation. 
void dsp_block(arma::vec& signal_1, arma::vec& signal_2, arma::mat& output_statistics)
{
  libeemd_error_code err_signal_1, err_signal_2;
 
  // We have only two phases as input signals from the motors.
  double* phase_1_input_vector = (double*) malloc(N*sizeof(double));
  double* phase_2_input_vector = (double*) malloc(N*sizeof(double));
  memset(phase_1_input_vector, 0x00, N*sizeof(double));
  memset(phase_2_input_vector, 0x00, N*sizeof(double));

  size_t M = emd_num_imfs(N);
  double* output_phase_1 = (double*)malloc(M*N*sizeof(double));
  double* output_phase_2 = (double*)malloc(M*N*sizeof(double));

  // Assign the signal to the C-style array, and do the computation on 
  // C style. The reason for this is that the EMD library is written in C99
  // I had to modify some part of the library too to make compile with g++
  // In essence, the IMF code is the same.
  // // I know this loop is stupid and can be avoided, I do remember that we can point
  // out a memory location and get a pointer from their directly from arma to anything 
  // else. I need to check how to do it.
  for (size_t i=0; i < N; ++i)
  {
    phase_1_input_vector[i] = signal_1(i);
    phase_2_input_vector[i] = signal_2(i);
  }

  // Run CEEMDAN
  err_signal_1 = ceemdan(phase_1_input_vector, N, output_phase_1, M, ensemble_size, noise_strength, S_number, num_siftings, rng_seed);
  if (err_signal_1 != EMD_SUCCESS)
  {
    emd_report_if_error(err_signal_1);
    exit(1);
  }

  err_signal_2 = ceemdan(phase_2_input_vector, N, output_phase_2, M, ensemble_size, noise_strength, S_number, num_siftings, rng_seed);
  if (err_signal_2 != EMD_SUCCESS)
  {
    emd_report_if_error(err_signal_2);
    exit(1);
  }
  // Write output to file
  // First write the signals it self as the first line of the file
  // Second write the IMFs from the output variable to the same file
  // Finally the file contains the original signals and the IMFS and the res.

  // The output to a file is not necessary for now, this can be removed later.
  // Keep it now, it is good for debugging.

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
  printf("Done!\n");
  // Clean the memory before losing all the heap.
  // fclose(fp);
  free(phase_1_input_vector); phase_1_input_vector = NULL;
  free(phase_2_input_vector); phase_2_input_vector = NULL;
  free(output_phase_1); output_phase_1 = NULL;
  free(output_phase_2); output_phase_2 = NULL;
}


// This function should prepare the signals and cut it into 15ms subsequent signals.
void prepare_signals()
{

  
}

int main(void)
{
  std::vector<arma::mat> dataset;
  load_all_datasets(dataset, 0);

  

  

}
