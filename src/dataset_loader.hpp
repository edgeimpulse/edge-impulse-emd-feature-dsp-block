#include <armadillo>
#include <string>

class DataLoader
{
  public:

  DataLoader(std::string path) : path(path)
  {}

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
            std::cerr << "The path for the classses is not correct, please fix the path and try again." 
                      << std::endl;
            exit(0);
          }
          datasets.push_back(file);
        }
      }
    } 
    else // parse class by number the dataset will be related to a specific class
    {
      for (size_t i = 1; i < 9; ++i)
      {
        bool state =
        file.load("../../datasets/raw_datasets/Sensorless_Drive_Diagnosis_Data_Set/class"
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
  }

  private:
  std::string path;

};


