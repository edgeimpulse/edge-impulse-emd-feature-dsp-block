# EMD Features Custom DSP Block

## Overview
This repository impelements the EMD based feature extranction algorithm described in "Feature Extraction and Reduction Applied to Sensorless Drive Diagnosis" [1](#citations) as an edge-impulse custom dsp block.

### :construction: This repository is under construction :construction:
This is currently a work in progress, implementation details and features may change and evolve in future revisions. Only minimal functional testing has been performed. 

## Usage
This DSP block is primarily intended to explore processing MCSA fault detection and predictive maintenance use cases for embedded systems via the [Dataset for Sensorless Drive Diagnosis](https://zenodo.org/record/35577#.XNA8mI4zZPY) [2](#citations)

A work-in-progress edge-impulse project utilizing this block to perform error classification may be found at the link below:
(https://studio.edgeimpulse.com/public/41083/latest)

This DSP block is currently not hosted anywhere at the moment, and must be hosted locally by users as a result. For information on how to host this block and use it in an edge impulse project, see the docs: [Building custom processing blocks](https://docs.edgeimpulse.com/docs/custom-blocks).

## Implementation details 

This project uses this [emd library's](https://emd.readthedocs.io/en/stable/) implementation to generate the first three IMFs and residuals of a provided window of time-series data. From these 6 IMF & residual functions, the mean, skewness, kurtosis, and normalized error values are computed and stored. 

For an excellent overview of what EMD is and the high level motivation for using it, see [1](#citations) and the [emd library docs](https://emd.readthedocs.io/en/stable/emd_tutorials/index.html)

## Citations
[1] Bator, Martyna & Dicks, Alexander & Mönks, Uwe & Lohweg, Volker. (2012). Feature Extraction and Reduction Applied to Sensorless Drive Diagnosis. 10.13140/2.1.2421.5689. 

[2] F. Paschke, C. Bayer, M. Bator, U. Mönks, A. Dicks, O. Enge-Rosenblatt, and V. Lohweg, “Sensorlose Zustandsüberwachung an Synchronmotoren,” in Proceedings 23. Workshop Computational Intelligence, Karlsruhe: KIT Scientific Publishing, 2013, pp. 211–225. [2] C. Bayer, M. Bator, U. Mönks, A. Dicks, O. Enge-Rosenblatt, and V. Lohweg, “Sensorless Drive Diagnosis Using Automated Feature Extraction, Significance Ranking and Reduction,” in 18th IEEE Int. Conf. on Emerging Technologies and Factory Automation (ETFA 2013): IEEE, 2013, pp. 1–4.
