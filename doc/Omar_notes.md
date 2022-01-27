## Omar notes' on Motor Current Signature Analysis (MCSA)

These are my notes on the project, trying to analyze and breif what have been done
in the past and what we need to do in the future.

### Current Status of the project:

There are three papers related to this project:

* The first article describes the data set processing:

[Feature Extraction and Reduction Applied to Sensorless Drive Diagnosis.]() Bator, Martyna & Dicks, Alexander & Mönks, Uwe & Lohweg, Volker. (2012).

* The second article is a berif of the first article:

[Sensorless Drive Diagnosis Using Automated Feature Extraction, Significance Ranking and Reduction]() Bayer, M. Bator, U. Mönks, A. Dicks, O. Enge-Rosenblatt, and V. Lohweg. 2013.

* The third articles in evaluating several machine learning methods: 

[Evaluation of Machine Learning for Sensorless Detection and Classification of Faults in Electromechanical Drive Systems](https://www.researchgate.net/publication/344492789_Evaluation_of_Machine_Learning_for_Sensorless_Detection_and_Classification_of_Faults_in_Electromechanical_Drive_Systems) T. Gruner et al. 2020.

* Fourh article that is using the same dataset (not the same authors):

[Group Sparse Regularization for Deep Neural Networks](https://arxiv.org/abs/1607.00485) S. Scardapane, D.Comminiello, A. Hussain, A. Uncini. 2016 


### The datasets

The project uses 2 datasets: 

* The first dataset is the raw dataset from the German Autmatica
https://zenodo.org/record/35577#.YfGYWIrLdhF

This dataset has been analyzed and the features has been extracted to created the second dataset.

* The second dataset is the proccessed dataset uploaded by the authors to the dataset portal:
https://archive.ics.uci.edu/ml/datasets/dataset+for+sensorless+drive+diagnosis#

This dataset contains 11 classes, 58K feature vector each has 48 dimensions.

* Dataset separation rate 70/30
* Data set is uniformly distributed over both classes and training and testing sets.
* Data set normalization is important for Knn and SVM, no different in results is shown in tree based methods


### The current methods 

EMD
IMF
??


### The tests that David has done:

#### The first test:

D1: The first analysis related to this project is using the raw dataset and applying the same exact method to 
extract features from the raw dataset using the DSP block that contains our implementation of the EMD

The dataset have been treated locally and uploaded to the studio?? ( not sure to be detailed)

MFE Processing block for audio is used in this project ? why 

The training method used is 1D Conv NN.

ANN Hyper parameters:
* Epochs = 1000
* Input layer (4800)
* Reshape layer (80)
* 1D Conv (32, 5 kernel, 2 layers)
* Flatten layer
* Dense layer
* Output layer 16


Useful links:

* Project link in the studio:  https://studio.edgeimpulse.com/public/41083/latest

The DSP block that we have implementated can be found here:
* https://github.com/Dasch0/edge-impulse-emd-feature-dsp-block


#### The second test

D2: The second analysis is done directly using the feature extracted dataset that is done by the authors and published on then UCI website.

The training method used is ANN Feed forward network.

ANN Hyper parameters:
* Epochs = 200
* layers (48, 96, 48, 24, 12)


Useful links:
* Project link in the studio: https://studio.edgeimpulse.com/studio/38818


### Authors results

* All classification results acheive accuracy of > 98 percent
* ANN hyper paramaeters are well mentioned and explained.
 

Authors results:

|Project     | Method used | Dataset used | Accuracy (Training) | Acurracy (Testing) |
--------------------------------------------------------------------------------------
| 2020  Paper | Knn     |    M(RFE25)     |   100              |  99.94              |
|             | SVM     |    M(RFE25)     |   100              |  99.66              |
|             | XGBoost |       M         |   100              |  99.84              |
|             | Random Foreset|  M        |   100              |  99.92              |
|             |  ANN 3  |        ?        |   99.91            |  99.64              | 
|             |  ANN 20 |        ?        |   99.45            |  99.36              |
------------------------------------------------------------------------------------


Our results: 


|Project     | Method used | Dataset used | Accuracy (Training) | Acurracy (Testing) |
--------------------------------------------------------------------------------------
| D1         | 1D Conv NN  |    Raw + DSP = ?  |   97.9          |  87.00            |
| D2(v1)     |   FFN       |    M (EI treatement)|  98.5         |                   |
| D2(v2)     |   SVM       |    M   EI          |   66.40        |                   |
--------------------------------------------------------------------------------------




