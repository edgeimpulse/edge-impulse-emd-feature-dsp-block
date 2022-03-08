


#ifndef DATALOADER_HPP
#define DATALOADER_HPP

#include <armadillo>
#include <string>
#include <vector>

class DataLoader
{

public:

  DataLoader(std::string path,
             std::vector<arma::mat>& datasets)
    : path(path)
    , datasets(datasets)
  {}

  // This file is only necessary to parse the original raw datasets
  // and load all of them into one matrix and then save the matrix to 
  // We have 11 classes, which means 11 files in totals as a results.

  void load_class(int class_num)
  {
    arma::mat file;
    for (size_t i = 1; i < 9; ++i)
    {
      bool state =
      file.load(path + "/class"
      + std::to_string(class_num) + "/class" + std::to_string(class_num) + "_Parameterset" 
      + std::to_string(i) + ".txt");
      std::cout << "Loading State: "<< state << " Loading class: "<< class_num << " file num: " << i << std::endl;
      if (state == false)
      {
        std::cerr << "The path for the classses is not correct, please fix the path and try again." 
                  << std::endl;
        exit(0);
      }
      datasets.push_back(file);
    }
  }

  void load_all()
  {
    arma::mat file;
    for (size_t j = 1; j < 12; ++j)
    {
      for (size_t i = 1; i < 9; ++i)
      {
        bool state =
        file.load(path+ "/class"
        + std::to_string(j) + "/class" + std::to_string(j) + "_Parameterset" 
        + std::to_string(i) + ".txt");
        std::cout << "Loading State: "<< state << " Loading class number: " << j << " file num: " << i << std::endl;
        if (state == false)
        {
          std::cerr << "The path for the classses is not correct, please fix the path and try again." 
                    << std::endl;
          exit(0);
        }
        datasets.push_back(file);
      }
    }
  }
  
private:
  std::string path;
  std::vector<arma::mat>& datasets
};

#endif // DATALOADER_HPP
