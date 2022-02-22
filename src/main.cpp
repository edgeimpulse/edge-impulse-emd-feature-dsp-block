#include "CLI11.hpp"
#include "dataset_loader.hpp"
#include "dsp.hpp"



// This software is composed of 3 parts
// The first part is the data loader which loads all the dataset into a one
// giant std vector.
// Second part is the EMD computation part which contains the C
// code that is used to compute the IMF
// The third part statistical part which contains the computation
// of the mean, std dev, skewness, etc...
// Finally the fourth part, saving all of these extracted features to a
// file locally. 

int main(int argc, char** argv)
{
  CLI::App app{"This software contains the necessary implementation to convert the raw MCSA dataset"
               " into a usable statistical feature dataset using the CEEMDAN methods.\n" 
               "This software can convert the entire dataset published by the Automatica platform, or"
               " only a specific raw dataset that is obtained from the motor directly.\n"
               "This software can be considered as a DSP block that extract features from the raw motors signals.\n"};

  std::string path;
  app.add_option("-p", path, "Enter the full path to the entire dataset. Usually this folder contains a set of classes folders.");
  
  CLI11_PARSE(app, argc, argv);
  
  std::vector<arma::mat> dataset;
  arma::mat features_matrix;

  if (!path.empty())
  {
    DataLoader data(path);
    data.load_all_datasets(dataset, 0);
  }

  create_feature_matrix(dataset, features_matrix);
  //Verify that the matrix is good.
  features_matrix = features_matrix.t();
  features_matrix.save("extracted_features.csv", arma::csv_ascii);
}


// Write the code of inference,
// Divide the code into a set of classes
// Add more docs related to each function

