#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <gsl/gsl_math.h>
#include <armadillo>

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
    for (size_t j = 1; j < 12; ++j)
    {
      for (size_t i = 1; i < 9; ++i)
      {
        bool state =
        file.load("../../datasets/raw_datasets/Sensorless_Drive_Diagnosis_Data_Set/class"
        + std::to_string(j) + "/class" + std::to_string(j) + "_Parameterset" 
        + std::to_string(i) + ".txt");
        std::cout << "Loading State: "<< state << " Loading class number: " << j << " file num: " << i << std::endl;
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
      std::cout << "Loading State: "<< state << " Loading class: "<< class_num << " file num: " << i << std::endl;
      if (state == false)
      {
        //see later what to do
      }
      datasets.push_back(file);
    }
  }
}

void Skewness(const arma::mat& imfs, const arma::vec& mean_signal, const arma::vec& stddev_signal, arma::vec& skewness)
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

void Kurtoise(const arma::mat& imfs, const arma::vec& mean_signal, const arma::vec& stddev_signal, arma::vec& kurtoise)
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
void dsp_block(arma::vec& signal_1, arma::vec& signal_2, arma::vec& output_statistics_signal_1, arma::vec& output_statistics_signal_2)
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
  
  // kurtoise_signal_1.print("Kurtoise");
  // kurtoise_signal_2.print("Kurtoise");

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

// This function should prepare the signals and cut it into 15ms subsequent signals.
void prepare_signals(arma::mat& dataset, arma::vec& signal_1, arma::vec& signal_2, int which_subsignals)
{
  for(size_t i = 0; i < 1500; ++i)
  {
    signal_1.at(i) = dataset.at((which_subsignals * 1500) + i, 1);
    signal_2.at(i) = dataset.at((which_subsignals * 1500) + i, 2);
  }
}

void create_feature_matrix_per_class(std::vector<arma::mat>& dataset, arma::mat& features_matrix, int class_num)
{
  double size = dataset.at(0).n_rows;
  std::cout << "size of the first dataset " << size << std::endl;
  size_t num_of_subsignals = size / 1500;
  std::cout << "num_of_subsignals " << num_of_subsignals << std::endl;
  std::cout << "Size of feature matrix (above): " << arma::size(features_matrix) << std::endl;

  arma::vec signal_1(1500, arma::fill::none);
  arma::vec signal_2(1500, arma::fill::none);

  // If we are going to iterate we need to call everything inside this loop
  arma::vec features_cols;
  for (size_t j = 0; j < 8; ++j)
  {
    for (size_t i = 0; i < num_of_subsignals; ++i)
    { 
      // j + (class_num * 8) is to access the index of the matrix that contains the 
      // files that has been loaded to each matrix, since we have only 8 files
      // per class.
      prepare_signals(dataset.at(j + ((class_num -1) * 8)), signal_1, signal_2, i);
   
      // Just for debugging.
      // signal_1.print("1");
      // signal_2.print("2");

      // Testing the DSP block
      arma::vec output_statistics_signal_1, output_statistics_signal_2;
      dsp_block(signal_1, signal_2, output_statistics_signal_1, output_statistics_signal_2);

      // output_statistics_signal_1.print("1 stats: ");
      // output_statistics_signal_2.print("2 stats: ");
      features_cols = arma::join_cols(output_statistics_signal_1, output_statistics_signal_2);
      arma::rowvec class_id( 1 /* this should be 8 * num_of_subsignals */, arma::fill::value(class_num));
      features_cols.insert_rows(features_cols.n_rows, class_id);
      // features_cols.print("Features cols:");
      features_matrix.insert_cols(features_matrix.n_cols, features_cols);
    }
  }
  std::cout << "Size of feature matrix: " << arma::size(features_matrix) << std::endl;

  // Adding the row at this level is causing an issue as the number of cols has increased.
  // therefore the next iteration will fail as the number of cols has increased by one

  // features_matrix.print("features: "); // Keep for debugging
  //features_matrix = features_matrix.t();
}

void create_feature_matrix(std::vector<arma::mat>& dataset, arma::mat& feature_matrix)
{
  for (size_t i = 1; i < 12; ++i)
  {
    create_feature_matrix_per_class(dataset, feature_matrix, i);
  }
}

int main(void)
{
  std::vector<arma::mat> dataset;
  arma::mat features_matrix;
  load_all_datasets(dataset, 0);
  create_feature_matrix(dataset, features_matrix);
  //Verify that the matrix is good.
  features_matrix = features_matrix.t();
  features_matrix.save("extracted_features.csv", arma::csv_ascii);
}
