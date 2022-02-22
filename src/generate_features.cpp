#include "generate_features.hpp"

FeatureGenerator::FeatureGenerator();

void FeatureGenerator::prepare_signals(int which_subsignals)
{
  for(size_t i = 0; i < 1500; ++i)
  {
    signal_1.at(i) = dataset.at((which_subsignals * 1500) + i, 1);
    signal_2.at(i) = dataset.at((which_subsignals * 1500) + i, 2);
  }
}

void FeatureGenerator::generate_feature_matrix_per_class(int class_num)
{
  double size = dataset.at(0).n_rows;
  std::cout << "size of the first dataset " << size << std::endl;
  size_t num_of_subsignals = size / 1500;
  std::cout << "num_of_subsignals " << num_of_subsignals << std::endl;
  std::cout << "Size of feature matrix (above): " << arma::size(features_matrix) << std::endl;

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
}

void FeatureGenerator::generate_feature_matrix()
{
  for (size_t i = 1; i < 12; ++i)
  {
    create_feature_matrix_per_class(dataset, feature_matrix, i);
  }
}

void FeatureGenerator::recursive_feature_elimination()
{
  /*
   * These are the set of features that have been selected by the RFE 
   * algorithm using the SVM kernel.
   * For the RFE code, please check the RFE repository.
   * 
   * These features provide a slightly increased accuracy, only one percent 
   * compared to the entire dataset, but it is recommended to use this one to 
   */
   
  // 7, 8, 9, 19, 20, 21, 24, 25, 26, 27, 28, 30, 31, 33, 34, 35, 36, 37, 38
  // 39, 47, 48, 49, 57, 58, 59, 61, 64, 65, 66, 67, 68, 69, 70, 71, 75, 76
  // 77, 78, 79.


}
