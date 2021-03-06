## Omar notes' on Motor Current Signature Analysis (MCSA)

These are my notes on the project, trying to analyze and brief what have been done
in the past and what we need to do in the future.

### Current Status of the project:

There are three papers related to this project:

* The first article describes the data set processing:

[Feature Extraction and Reduction Applied to Sensorless Drive Diagnosis.](https://www.researchgate.net/publication/264273485_Feature_Extraction_and_Reduction_Applied_to_Sensorless_Drive_Diagnosis) Bator, Martyna & Dicks, Alexander & Mönks, Uwe & Lohweg, Volker. (2012).

* The second article is a brief of the first article:

[Sensorless Drive Diagnosis Using Automated Feature Extraction, Significance Ranking and Reduction](https://www.researchgate.net/publication/261282496_Sensorless_drive_diagnosis_using_automated_feature_extraction_significance_ranking_and_reduction) Bayer, M. Bator, U. Mönks, A. Dicks, O. Enge-Rosenblatt, and V. Lohweg. 2013.

* The third articles in evaluating several machine learning methods: 

[Evaluation of Machine Learning for Sensorless Detection and Classification of Faults in Electromechanical Drive Systems](https://www.researchgate.net/publication/344492789_Evaluation_of_Machine_Learning_for_Sensorless_Detection_and_Classification_of_Faults_in_Electromechanical_Drive_Systems) T. Gruner et al. 2020.

* Fourth article that is using the same dataset (not the same authors):

[Group Sparse Regularization for Deep Neural Networks](https://arxiv.org/abs/1607.00485) S. Scardapane, D.Comminiello, A. Hussain, A. Uncini. 2016 

### The datasets

The project uses 2 datasets: 

* The first dataset is the raw dataset from the German Automatica

```
https://zenodo.org/record/35577#.YfGYWIrLdhF
```
This dataset has been analyzed and the features has been extracted to created the second dataset.

**Zenodo dataset details**
* First column is the time stamps
* Second column is the first phase
* Third column is the second phase
* Each file contains 10 secs of dataset.

* The second dataset is the processed dataset uploaded by the authors to the dataset portal:

```
https://archive.ics.uci.edu/ml/datasets/dataset+for+sensorless+drive+diagnosis#
```

This dataset contains 11 classes, 58K feature vector each has 48 dimensions.

* Dataset separation rate 70/30
* Data set is uniformly distributed over both classes and training and testing sets.
* Data set normalization is important for Knn and SVM, no different in results is shown in tree based methods


**UCI Dataset details**

```
Nr      statistic feature	phase	RES\IMF
1	mean			1	IMF1
2	mean			1	IMF2
3	mean			1	IMF3
4	mean			2	IMF1
5	mean			2	IMF2
6	mean			2	IMF3
7	mean			1	RES1
8	mean			1	RES2
9	mean			1	RES3
10	mean			2	RES1
11	mean			2	RES2
12	mean			2	RES3
13	standard deviation	1	IMF1
14	standard deviation	1	IMF2
15	standard deviation	1	IMF3
16	standard deviation	2	IMF1
17	standard deviation	2	IMF2
18	standard deviation	2	IMF3
19	standard deviation	1	RES1
20	standard deviation	1	RES2
21	standard deviation	1	RES3
22	standard deviation	2	RES1
23	standard deviation	2	RES2
24	standard deviation	2	RES3
25	skewness		1	IMF1
26	skewness		1	IMF2
27	skewness		1	IMF3
28	skewness		2	IMF1
29	skewness		2	IMF2
30	skewness		2	IMF3
31	skewness		1	RES1
32	skewness		1	RES2
33	skewness		1	RES3
34	skewness		2	RES1
35	skewness		2	RES2
36	skewness		2	RES3
37	kurtosis		1	IMF1
38	kurtosis		1	IMF2
39	kurtosis		1	IMF3
40	kurtosis		2	IMF1
41	kurtosis		2	IMF2
42	kurtosis		2	IMF3
43	kurtosis		1	RES1
44	kurtosis		1	RES2
45	kurtosis		1	RES3
46	kurtosis		2	RES1
47	kurtosis		2	RES2
48	kurtosis		2	RES3
49	Class-ID	 	 
-------------------------------------------------------------------------------------------------
The class IDs are the following: (SM=shaft misalignment, AI=axle inclination, BF=bearing failure)

ClassID		BF	AI	SM
 1		0	0	0
 2		0	0	1
 3		0	1	0
 4		0	1	1
 5		0	1	1
 6		1	0	1
 7		1	1	0
 8		1	1	1
 9		1	0	1
10		1	1	0
11		1	1	1

Class1 : Normal state (no failure in any module),
Class2: shaft failure, and so on. 
Class 4 and 5 are not the same, they differ in the angle of the axis inclination. The situation is similar for class8 and class11.
```

### Window Size estimation
* Each class from the raw dataset has 8 files.
* Each file contains dataset of 10 seconds.
* In total each class has 80 seconds datasets.
* In the UCI datset, each class has 5139 lines
* Which means that: each line represent the statistical features of 15 ms signals
* This deduced by 80/5130 = 0.015 sec -> 15 ms.
* The first data set is captured at 100 KHZ.
* Which means that 15 ms is equal to 1500 lines of the first dataset.
* As a result: each line of the UCI dataset is computed from 1500 lines of the raw dataset.

### Our Dataset generated from the C++ DSP
* The size of our dataset is 58608 line comapred to 58509 original one.
* Currently our dataset does not contains the Skewness and the kurtoise
* Our dataset contains 11 IMF instead of 3 IMF and res
* Our algorithm is using ceemdan instead of the original emd algorithm.
* Untill now the traning locally is not giving the extact same results achieved before,
* The objectives is to move forward to get the same or a similar results.
* The IMF and statistic features extractions is taking 36 hours of training.


### The current methods 

In order to extract features from the raw datasets, the authors are using the EMD (Empirical Mode Decomposition) method
This methods is usually applied on complex signals that have several frequencies. The EMD method allows to extract these frequencies as a 
set of functions know as Intrinsic Mode Functions (IMF). The higher mode number is the smaller frequency it contains.

Therefore, the sum of all the mode functions and the residual provides us back with the original signal.

```
f(t) = Sum(imf(i)) + res
```

**EMD Steps**:
1) Find the local extrema of the signal (Max and Min).
2) Fit an envelope on the all maxima and minima point that contains all the signal data.
3) Create a mean envelope from the min and max envelope E(mean)t = (E(up)t + E(low)t) \ 2.
4) Determine residual by subtracting the original signal from the mean envelope.
res = f(x) - E(mean)t
5) Check the stopping criteria.
6) Iterate until 5 stop the process.

Once the IMF is extracted, the authors partition the IMFS into a set of subsequent signals.
Then for each of subsequent, a set of statistical features are calculated.
The set of features that are calculated are the following:
1) Mean
2) Standard deviation
3) Skewness
4) Excess
5) Normalized error indicator.

The statistical feature are agglomerated for each subsequence to create a long feature vector related to each signal.
Therefore, a reduction of feature vector is required.

**Linear Discriminant Analysis (LDA)**


### The tests that David has done:

#### The first test:

D1: The first analysis related to this project is using the raw dataset and uploading the same
dataset directly to the studio, the MFE Processing block for audio is used in this project as a 
DSP block in order to treat the dataset.

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


The DSP block that we have implemented can be found here:
```
* https://github.com/Dasch0/edge-impulse-emd-feature-dsp-block
```
This DSP block tries to transform the raw dataset to look similar to UCI dataset.
This block has been used only locally.

#### The second test

D2: The second analysis is done directly using the feature extracted dataset that is done by the authors and published on then UCI website.

Each line of the dataset is treated as file and label deduced from the file name, no transformation block was required.

It is possible to extract the dataset from the studio back to Json

The training method used is ANN Feed forward network.


**Version 1 Hyper parameters**

ANN Hyper parameters:
* Epochs = 200
* layers (48, 96, 48, 24, 12)

**Version 2 Hyper parameters**
* SVM traditional ML, not supported yet.

**Version 3 Hyper parameters**

ANN Hyper parameters:
* Epochs = Unlimited, controlled by early stop at min loss (callback)
* Use tanh activation function, with softmax for last layer
* layers (48, 90, 120, 90, 12)


#### The third test

D3: The third test uses the dataset that generated by our DSP block that contains the implemetation
of an improved version of emd and implemented locally in C++.
This dataset contains all the IMFs generated from each signal instead of the first three IMF and RES.
The dataset have very similar size to the one generated by the authors.

* The feature vector of this dataset is double the feautre vector of the original dataset due to the new IMFs

**Version 1 Hyper parameters**
ANN Hyper parameters:
* Epochs = Unlimited, controlled by early stop at min loss (callback)
* Use tanh activation function, with softmax for last layer
* layers (80, 180, 240, 180, 12)
* Add three Dropout layer with 0.2 percentage of the dropout.
* Since the feature vector size is different, we have doubled the number of neurons in each hidden layer
* Achieved accuracy on the testing dataset is 91 percent on only 2000 epochs

**Version 2 Hyper parameters**
ANN Hyper parameters:
* Epochs = Unlimited, controlled by early stop at min loss on 200 (callback)
* Use tanh activation function, with softmax for last layer
* Layers (80, 180, 300, 180, 12)
* Add three Dropout layer with 0.2 percentage of the dropout.
* Since the feature vector size is different, we have doubled the number of neurons in each hidden layer
* Achieved accuracy on the testing dataset is 93 percent.

**Version 3 Hyper parameters**
ANN Hyper parameters:
* Epochs = Unlimited, controlled by early stop at min loss on 250 (callback)
* Use tanh activation function, with softmax for last layer
* Layers (80, 200, 350, 200, 12)
* Add three Dropout layer with 0.2 percentage of the dropout.
* Since the feature vector size is different, we have doubled the number of neurons in each hidden layer
* Achieved accuracy on the testing dataset is 94 percent.

Useful links:
* Project link in the studio: https://studio.edgeimpulse.com/studio/38818

### Authors results

* All classification results achieve accuracy of > 98 percent
* ANN hyper parameters are well mentioned and explained.
 

**Authors results**:

|Project     | Method used | Dataset used | Accuracy (Training)| Accuracy (Testing)  |
|------------|-------------|------------- |--------------------|-------------------- |
| 2020  Paper | Knn     |    M(RFE25)     |   100              |  99.94              |
| ML  Evaluation | SVM  |    M(RFE25)     |   100              |  99.66              |
|             | XGBoost |       M         |   100              |  99.84              |
|             | Random Forest |  M        |   100              |  99.92              |
|             |  ANN 3  |        ?        |   99.91            |  99.64              | 
|             |  ANN 20 |        ?        |   99.45            |  99.36              |


**Our results**: 

|Project     | Method used | Dataset used | Accuracy (Training) | Accuracy (Testing) |
|--------    |-------------|--------------|---------------------|--------------------|
| D1         | 1D Conv NN  |    Raw + DSP = ?  |   97.9          |  87.00            |
| D2(v1)     |   FFN       |    M (EI treatment)|  98.5          |  96.59            |
| D2(v2)     |   SVM       |    M   EI          |   66.40        |   ?               |
| D2(v3)     |   FFN       |    M   EI          |   98.7         |  96.69            |
| D3(v1)     |   FFN       |    Omar Datasets   |   88           |   91.2            |
| D3(v2)     |   FFN       |    Omar Datasets   |   82 N dropout |   93              |
| D3(v3)     |   FFN       |    Omar Datasets   |   84 N dropout |   94              |

### What do we need to optimize

**Short term objectives:**
* [ ] Accuracy that need to be at least 98 percent for testing datasets.
* [X] Start working on the DSP block in C++ for fast computation
* [X] Get the Jupyter notebook code and try it locally
* [X] Try with local tests with different ML frameworks.
* [ ] Use the RFE method to produce the 25 column datasets.
* [ ] Try Knn, SVM locally with the RFE25 dataset

**Long term objectives:**
* [X] We would like the run time to be fast, less care about memory
* [X] Objective to get the similar 48 feature that we already have in the dataset
* [X] Window size requires optimization too.
* [ ] Finally test the model on the Test bed 


