from dsp import get_next_imf
import emd
import numpy as np
from scipy import stats
import sys, time

BALL_BEARING_PERIOD = 0.022

# obtain a list of the mean values of each imf
def get_imf_mean(emd):
    return [np.mean(emd[0][0]), np.mean(emd[0][1]), np.mean(emd[0][2])]

# obtain a list of the mean values of each residual
def get_res_mean(emd):
    return [np.mean(emd[1][0]), np.mean(emd[1][1]), np.mean(emd[1][2])]

# obtain a list of the standard deviations values of each imf
def get_imf_stddev(emd):
    return [np.std(emd[0][0]), np.std(emd[0][1]), np.std(emd[0][2])]

# obtain a list of the standard deviations values of each residual
def get_res_stddev(emd):
    return [np.std(emd[1][0]), np.std(emd[1][1]), np.std(emd[1][2])]

# obtain a list of the skew of each imf
def get_imf_skew(emd):
    return [stats.skew(emd[0][0]), stats.skew(emd[0][1]), stats.skew(emd[0][2])]

# obtain a list of the skew of each residual
def get_res_skew(emd):
    return [stats.skew(emd[1][0]), stats.skew(emd[1][1]), stats.skew(emd[1][2])]

# obtain a list of the kurtosis of each imf
def get_imf_kurtosis(emd):
    return [stats.kurtosis(emd[0][0]), stats.kurtosis(emd[0][1]), stats.kurtosis(emd[0][2])]

# obtain a list of the kurtosis of each residual
def get_res_kurtosis(emd):
    return [stats.kurtosis(emd[1][0]), stats.kurtosis(emd[1][1]), stats.kurtosis(emd[1][2])]

# data storage for the EMD analysis of a single time series signal
# computes IMFs following the algorithm for Sensorless Drive Diagnosis
class EmdAnalysisData:
    fns: np.ndarray

    # initialize from 1d timeseries data buffer
    #
    # first axis is imf or res, second axis is emd iteration number, 3rd axis is the imf/res data
    def __init__(self, buf):
        self.fns = np.empty((2, 6, buf.size))
        self.fns[0][0] = get_next_imf(buf)
        self.fns[1][0] = buf - self.fns[0][0]
        self.fns[0][1] = get_next_imf(self.fns[1][0])
        self.fns[1][1] = self.fns[1][0] - self.fns[0][1]
        self.fns[0][2] = get_next_imf(self.fns[1][1])
        self.fns[1][2] = (self.fns[1][1] - self.fns[0][2])

    # grab a slice of the imf and residuals into equal sized slices
    def slice(self, slice_range):
        return self.fns[:,:,slice_range]

    
    # retrieve the next imf of the input signal. Use this on the prior computed imf or base signal
    def get_next_imf(self, x, zoom=None, sd_thresh=0.1):
        proto_imf = x.copy()  # Take a copy of the input so we don't overwrite anything
        continue_sift = True  # Define a flag indicating whether we should continue sifting
        niters = 0            # An iteration counter

        if zoom is None:
            zoom = (0, x.shape[0])

        # Main loop - we don't know how many iterations we'll need so we use a ``while`` loop
        while continue_sift:
            niters += 1  # Increment the counter

            # Compute upper and lower envelopes
            upper_env = emd.utils.interp_envelope(proto_imf, mode='upper')
            lower_env = emd.utils.interp_envelope(proto_imf, mode='lower')

            # Compute average envelope
            avg_env = (upper_env + lower_env) / 2

            # Should we stop sifting?
            stop, val = emd.sift.sd_stop(proto_imf-avg_env, proto_imf, sd=sd_thresh)

            # Remove envelope from proto IMF
            proto_imf = proto_imf - avg_env

            # and finally, stop if we're stopping
            if stop:
                continue_sift = False

        # Return extracted IMF
        return proto_imf

# Obtain and format statistical features of EMD analysis to match the expected format used by the 
# Dataset for Sensorless Drive Diagnosis
def get_features_from_emd_slice(emd0, emd1):
    flattened_features =        \
        get_imf_mean(emd0) +       \
        get_imf_mean(emd1) +       \
        get_res_mean(emd0) +       \
        get_res_mean(emd1) +       \
        get_imf_stddev(emd0) +     \
        get_imf_stddev(emd1) +     \
        get_res_stddev(emd0) +     \
        get_res_stddev(emd1) +     \
        get_imf_skew(emd0) +       \
        get_imf_skew(emd1) +       \
        get_res_skew(emd0) +       \
        get_res_skew(emd1) +       \
        get_imf_kurtosis(emd0) +   \
        get_imf_kurtosis(emd1) +   \
        get_res_kurtosis(emd0) +   \
        get_res_kurtosis(emd1)

    return flattened_features


# Process 2 phase current signal input into features. Data should be structured as a 2d numpy array
# Each axis of the input is a single current capture. The output is a Nx48 array.
# 
# N is the number of slices extracted from the signal. 48 is the number of statistical features
# computed for each slice. Each row of the output should be fed into the trained NN to run inference
# one time, as inference is performed on each slice individually
#
# For more details, see the citation [1] in the readme for the algorithm described by the Dataset for 
# Sensorless Drive Diagnosis
def process_current_signal_into_features(data: np.ndarray, sample_frequency_hz: float):
    # get the two AC current signals from the ndarray
    signal_ac0 = data[0]
    signal_ac1 = data[1]

    # equal length signals assumed later in feature extraction
    if signal_ac0.size != signal_ac1.size:
        print("feature processing failed, input current signals must be of equal length for each phase") 
        sys.exit(1)
    signal_len = signal_ac0.size

    # run EMD analysis on both input signals
    emd_ac0 = EmdAnalysisData(signal_ac0)
    emd_ac1 = EmdAnalysisData(signal_ac1)

    # slice EMD based on approximate ball bearing revolution period
    sample_period = 1.0 / sample_frequency_hz
    samples_per_revolution = int(BALL_BEARING_PERIOD / sample_period)
    
    # generate arrays of features for inference
    features = []

    for i in range(0, signal_len - samples_per_revolution, samples_per_revolution):
        features.append(get_features_from_emd_slice(
            emd_ac0.slice(range(i, i + samples_per_revolution)),
            emd_ac1.slice(range(i, i + samples_per_revolution))))

    return features

# Load a sample .txt file from the Dataset for Sensorless Drive Diagnosis and parse it into a 2d numpy array
def load_sample_from_fs(file_path):
    data = np.loadtxt(file_path, usecols=(1,2), unpack=True)
    return data

if __name__ == "__main__":
    print("loading sample...")
    data = load_sample_from_fs("class1_Parameterset1.txt")
    downsampled_data = data[:,::2]
    print("processing features...")
    start_time = time.perf_counter_ns()
    features = process_current_signal_into_features(downsampled_data, 10000)
    end_time = time.perf_counter_ns()
    duration = end_time - start_time

    print("duration (ns): ", duration)
    

