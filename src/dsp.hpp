#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <gsl/gsl_math.h>
#include "eemd.h"


class DSP
{
  public:

  DSP();

  void Skewness(const arma::mat& imfs, 
                const arma::vec& mean_signal, 
                const arma::vec& stddev_signal, 
                arma::vec& skewness)
  {
    double sum = 0;
    for (size_t i = 0; i < imfs.n_rows; ++i)
    {
      for (size_t j = 0; j < imfs.n_cols; ++j)
      {
        sum = (imfs.at(i, j) - mean_signal.at(i)) * (imfs.at(i, j) - mean_signal.at(i)) * (imfs.at(i, j) - mean_signal.at(i));
      }
      skewness.at(i) = sum / (imfs.row(i).n_elem * stddev_signal.at(i) * stddev_signal.at(i) * stddev_signal.at(i));
    }
  }

  void Kurtoise(const arma::mat& imfs, 
                const arma::vec& mean_signal, 
                const arma::vec& stddev_signal, 
                arma::vec& kurtoise)
  {
    double sum = 0;
    for (size_t i = 0; i < imfs.n_rows; ++i)
    {
      for (size_t j = 0; j < imfs.n_cols; ++j)
      {
        sum = (imfs.at(i, j) - mean_signal.at(i)) * (imfs.at(i, j) - mean_signal.at(i)) * (imfs.at(i, j) - mean_signal.at(i)) * (imfs.at(i, j) - mean_signal.at(i));
      }
      kurtoise.at(i) = (sum / (imfs.row(i).n_elem * stddev_signal.at(i) * stddev_signal.at(i) * stddev_signal.at(i) * stddev_signal.at(i))) - 3;
    }
  }

  // This function constitutes the main functionality of
  // computing the IMF of the signals.
  // The signal matrix or vector, should be cut to 1500 and prepared for 
  // the dsp. This block does not handle any kind of data preparation. 
  void dsp_block(arma::vec& signal_1, 
                 arma::vec& signal_2, 
                 arma::vec& output_statistics_signal_1, 
                 arma::vec& output_statistics_signal_2)
  {
    libeemd_error_code err_signal_1, err_signal_2;
   
    // We have only two phases as input signals from the motors.
    double* phase_1_input_vector = (double*) malloc(N*sizeof(double));
    double* phase_2_input_vector = (double*) malloc(N*sizeof(double));
    memset(phase_1_input_vector, 0x00, N*sizeof(double));
    memset(phase_2_input_vector, 0x00, N*sizeof(double));

    size_t num_imfs = emd_num_imfs(N);
    double* output_phase_1 = (double*)malloc(num_imfs*N*sizeof(double));
    double* output_phase_2 = (double*)malloc(num_imfs*N*sizeof(double));

    // Assign the signal to the C-style array, and do the computation on 
    // C style. The reason for this is that the EMD library is written in C99
    // I had to modify some part of the library too to make compile with g++
    // In essence, the IMF code is the same.
    // I know this loop is stupid and can be avoided, I do remember that we can point
    // out a memory location and get a pointer from their directly from arma to anything 
    // else. I need to check how to do it.
    for (size_t i = 0; i < N; ++i)
    {
      phase_1_input_vector[i] = signal_1(i);
      phase_2_input_vector[i] = signal_2(i);
    }

    // Run CEEMDAN
    err_signal_1 = ceemdan(phase_1_input_vector, N, output_phase_1, num_imfs, ensemble_size, noise_strength, S_number, num_siftings, rng_seed);
    if (err_signal_1 != EMD_SUCCESS)
    {
      emd_report_if_error(err_signal_1);
      exit(1);
    }

    err_signal_2 = ceemdan(phase_2_input_vector, N, output_phase_2, num_imfs, ensemble_size, noise_strength, S_number, num_siftings, rng_seed);
    if (err_signal_2 != EMD_SUCCESS)
    {
      emd_report_if_error(err_signal_2);
      exit(1);
    }

    arma::mat output_arma_signal_1(num_imfs, N, arma::fill::none);
    arma::mat output_arma_signal_2(num_imfs, N, arma::fill::none);

    for (size_t i = 0; i < num_imfs; ++i)
    {
      for (size_t j = 0; j < N; ++j)
      {
        output_arma_signal_1.at(i, j) = output_phase_1[i * N + j];
        output_arma_signal_2.at(i, j) = output_phase_2[i * N + j];    
      } 
    }

    // Debugging
    // output_arma_signal_1.save("signal_1", arma::csv_ascii);
    // output_arma_signal_2.save("signal_2", arma::csv_ascii);

    arma::vec output_mean_signal_1 = arma::mean(output_arma_signal_1, 1); // Check the data order
    arma::vec output_mean_signal_2 = arma::mean(output_arma_signal_2, 1); // Check the data order
    arma::vec output_stddev_signal_1 = arma::stddev(output_arma_signal_1, 0, 1); // Check the data order
    arma::vec output_stddev_signal_2 = arma::stddev(output_arma_signal_2, 0, 1); // Check the data order

    arma::vec skewness_signal_1(arma::size(output_mean_signal_1), arma::fill::none);
    arma::vec skewness_signal_2(arma::size(output_mean_signal_2), arma::fill::none);

    arma::vec kurtoise_signal_1(arma::size(output_mean_signal_1), arma::fill::none);
    arma::vec kurtoise_signal_2(arma::size(output_mean_signal_2), arma::fill::none);

    Skewness(output_arma_signal_1, output_mean_signal_1, output_stddev_signal_1, skewness_signal_1);
    Skewness(output_arma_signal_2, output_mean_signal_2, output_stddev_signal_2, skewness_signal_2);

    Kurtoise(output_arma_signal_1, output_mean_signal_1, output_stddev_signal_1, kurtoise_signal_1);
    Kurtoise(output_arma_signal_2, output_mean_signal_2, output_stddev_signal_2, kurtoise_signal_2);

    // skewness_signal_1.print("Skewness");
    // skewness_signal_2.print("Skewness");

    arma::vec sk_kur_signal_1 = arma::join_cols(skewness_signal_1, kurtoise_signal_1);
    arma::vec sk_kur_signal_2 = arma::join_cols(skewness_signal_2, kurtoise_signal_2);

    arma::vec mean_stddev_signal_1 = arma::join_cols(output_mean_signal_1, output_stddev_signal_1);
    arma::vec mean_stddev_signal_2 = arma::join_cols(output_mean_signal_2, output_stddev_signal_2);

    output_statistics_signal_1 = arma::join_cols(mean_stddev_signal_1, sk_kur_signal_1);
    output_statistics_signal_2 = arma::join_cols(mean_stddev_signal_2, sk_kur_signal_2);

    free(phase_1_input_vector);
    phase_1_input_vector = NULL;

    free(phase_2_input_vector); 
    phase_2_input_vector = NULL;

    free(output_phase_1); 
    output_phase_1 = NULL;

    free(output_phase_2); 
    output_phase_2 = NULL;
  }

  private:
    const double pi = M_PI;
    const size_t ensemble_size = 500;
    const unsigned int S_number = 4;
    const unsigned int num_siftings = 50;
    const double noise_strength = 0.02;
    const unsigned long int rng_seed = 0;
    const size_t N = 1500; // Window size is equal to 15 ms
};
