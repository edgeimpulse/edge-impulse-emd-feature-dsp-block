#include <armadillo>

int main()
{
  arma::mat dataset;
  dataset.load("dataset.csv");
  for (size_t i = 0; i < dataset.n_rows; ++i)
  {  
    arma::rowvec feature_vector =  dataset.row(i);
    std::string name = "class" + std::to_string(feature_vector.at(feature_vector.n_elem - 1)) + ".__" + std::to_string(i);
    feature_vector.shed_col(feature_vector.n_elem - 1);
    feature_vector.save(name, arma::csv_ascii);
  }
}
