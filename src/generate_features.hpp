/*
 * Edge Impulse DSP block for the MCSA
 * Copyright (c) 2022 EdgeImpulse Inc.
 */
 
#ifndef FEATURE_GENERATOR_HPP
#define FEATURE_GENERATOR_HPP

class FeatureGenerator
{

public:

  FeatureGenerator();

  // This function should prepare the signals and cut it into 15ms subsequent signals.
  void prepare_signals(arma::vec& signal_1, arma::vec& signal_2, int which_subsignals);

  void generate_feature_matrix_per_class(int class_num);

  void generate_feature_matrix();
  
  void recursive_feature_elimination();

private:
  std::vector<arma::mat>& dataset;
  arma::mat& features_matrix;
  arma::mat& rfe_features_matrix;
  arma::vec signal_1(1500, arma::fill::none);
  arma::vec signal_2(1500, arma::fill::none);
};

#endif // FEATURE_GENERATOR_HPP
