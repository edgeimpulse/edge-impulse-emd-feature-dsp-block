#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <gsl/gsl_math.h>

#include <armadillo>

const double pi = M_PI;

extern "C" {
#include "eemd.h"
}

const size_t ensemble_size = 500;
const unsigned int S_number = 4;
const unsigned int num_siftings = 50;
const double noise_strength = 0.02;
const unsigned long int rng_seed = 0;
const char outfile[] = "trial.out";

const size_t N = 1500;

#define __restrict__ restrict

int main(void)
{
  libeemd_error_code err;
  double* input_vector = (double*) malloc(N*sizeof(double));
  memset(input_vector, 0x00, N*sizeof(double));
  size_t M = emd_num_imfs(N);
  double* outp = (double*)malloc(M*N*sizeof(double));

  arma::mat phase1;
  phase1.load("../../datasets/raw_datasets/Sensorless_Drive_Diagnosis_Data_Set/class1/class1_Parameterset1.txt");

  for (size_t i=0; i < N; ++i)
  {
    input_vector[i]= phase1(1, i);
  }

  // Run CEEMDAN
  err = ceemdan(input_vector, N, outp, M, ensemble_size, noise_strength, S_number, num_siftings, rng_seed);
  if (err != EMD_SUCCESS)
  {
    emd_report_if_error(err);
    exit(1);
  }

  // Write output to file
  FILE* fp = fopen(outfile, "w");
  for (size_t j=0; j<N; j++)
  {
    fprintf(fp, "%f ", input_vector[j]);
  }
  fprintf(fp, "\n");
  for (size_t i=0; i<M; i++)
  {
    for (size_t j=0; j<N; j++)
    {
      fprintf(fp, "%f ", outp[i*N+j]);
    }
    fprintf(fp, "\n");
  }
  printf("Done!\n");
  // Cleanup
  fclose(fp);
  free(input_vector); input_vector = NULL;
  free(outp); outp = NULL;
}
