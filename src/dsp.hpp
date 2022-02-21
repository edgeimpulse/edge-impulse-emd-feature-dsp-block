#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <gsl/gsl_math.h>
#include "eemd.h"

class DSP
{
  public:

  DSP(arma::vec& signal_1, 
      arma::vec& signal_2,
      arma::vec& output_statistics_signal_1
      arma::vec& output_statistics_signal_2);

  void Skewness(const arma::mat& imfs, 
                const arma::vec& mean_signal, 
                const arma::vec& stddev_signal, 
                arma::vec& skewness)

  void Kurtoise(const arma::mat& imfs, 
                const arma::vec& mean_signal, 
                const arma::vec& stddev_signal, 
                arma::vec& kurtoise)

  // This function constitutes the main functionality of
  // computing the IMF of the signals.
  // The signal matrix or vector, should be cut to 1500 and prepared for 
  // the dsp. This block does not handle any kind of data preparation. 
  void run_dsp();

  private:
    const double pi = M_PI;
    const size_t ensemble_size = 500;
    const unsigned int S_number = 4;
    const unsigned int num_siftings = 50;
    const double noise_strength = 0.02;
    const unsigned long int rng_seed = 0;
    const size_t window_size = 1500; // Window size is equal to 15 ms
    
    arma::vec& signal_1, 
    arma::vec& signal_2, 
    arma::vec& output_statistics_signal_1, 
    arma::vec& output_statistics_signal_2
};
