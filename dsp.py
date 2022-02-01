import numpy as np
from scipy import signal 
import emd
from scipy import stats

# helper function to flatten lists
def flatten(input):
    new_list = []
    for i in input:
        for j in i:
            new_list.append(j)
    return new_list

def axes_to_list(axes_data: dict) -> list:
    """helper method to convert a dict of sensor axis graphs to a 2d array for graphing
    """
    axes_tuples = axes_data.items()
    axes_list = [axes[1].tolist() for axes in axes_tuples]

    return axes_list

# retrieve the next imf of the input signal. Use this on the prior computed imf or base signal
def get_next_imf(x, zoom=None, sd_thresh=0.1):
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
        avg_env = (upper_env+lower_env) / 2

        # Should we stop sifting?
        stop, val = emd.sift.sd_stop(proto_imf-avg_env, proto_imf, sd=sd_thresh)

        # Remove envelope from proto IMF
        proto_imf = proto_imf - avg_env

        # and finally, stop if we're stopping
        if stop:
            continue_sift = False

    # Return extracted IMF
    return proto_imf

def frequency_domain_graph_y(sampling_freq, lenX):
    N = lenX
    T = 1 / sampling_freq
    freq_space = np.linspace(0.0, 1.0/(2.0*T), N//2)
    return freq_space.tolist()

def generate_features(draw_graphs, raw_data, axes, sampling_freq, **kwargs):
    # features is a 1D array, reshape so we have a matrix with one raw per axis
    raw_data = raw_data.reshape(int(len(raw_data) / len(axes)), len(axes))

    # results are the first 3 imfs and their residuals
    # storing as a dict in [imf][axis] order to match format expected by dsp server graphs
    result_functions = {'imf0' : {}, 'imf1' : {}, 'imf2' : {}, 'res0' : {}, 'res1' : {}, 'res2' : {}}

    # split out the data from all axes
    for ax in range(0, len(axes)):
        X = []
        for ix in range(0, raw_data.shape[0]):
            X.append(raw_data[ix][ax])

        # X now contains only the current axis
        fx = np.array(X)

        imf = []
        res = []

        # get first imfs and residuals
        result_functions['imf0'][axes[ax]] = get_next_imf(fx)
        result_functions['res0'][axes[ax]] = fx - result_functions['imf0'][axes[ax]]
        result_functions['imf1'][axes[ax]] = get_next_imf(result_functions['res0'][axes[ax]])
        result_functions['res1'][axes[ax]] = result_functions['res0'][axes[ax]] - result_functions['imf1'][axes[ax]]
        result_functions['imf2'][axes[ax]] = get_next_imf(result_functions['res1'][axes[ax]])
        result_functions['res2'][axes[ax]] = result_functions['res1'][axes[ax]] - result_functions['imf2'][axes[ax]]

    # compute statistical features for imfs and residuals
    # features is a fixed size 1x48 dimension list matching order of Sensorless Diagnosis dataset
    features = []
    # mean
    for ax in range(0, len(axes)):
        features.append(np.mean(result_functions['imf0'][axes[ax]]))
        features.append(np.mean(result_functions['imf1'][axes[ax]]))
        features.append(np.mean(result_functions['imf2'][axes[ax]]))
    for ax in range(0, len(axes)):
        features.append(np.mean(result_functions['res0'][axes[ax]]))
        features.append(np.mean(result_functions['res1'][axes[ax]]))
        features.append(np.mean(result_functions['res2'][axes[ax]]))
    # stddev
    for ax in range(0, len(axes)):
        features.append(np.std(result_functions['imf0'][axes[ax]]))
        features.append(np.std(result_functions['imf1'][axes[ax]]))
        features.append(np.std(result_functions['imf2'][axes[ax]]))
    for ax in range(0, len(axes)):
        features.append(np.std(result_functions['res0'][axes[ax]]))
        features.append(np.std(result_functions['res1'][axes[ax]]))
        features.append(np.std(result_functions['res2'][axes[ax]]))
    # skew
    for ax in range(0, len(axes)):
        features.append(stats.skew(result_functions['imf0'][axes[ax]]))
        features.append(stats.skew(result_functions['imf1'][axes[ax]]))
        features.append(stats.skew(result_functions['imf2'][axes[ax]]))
    for ax in range(0, len(axes)):
        features.append(stats.skew(result_functions['res0'][axes[ax]]))
        features.append(stats.skew(result_functions['res1'][axes[ax]]))
        features.append(stats.skew(result_functions['res2'][axes[ax]]))
    # kurtosis
    for ax in range(0, len(axes)):
        features.append(stats.kurtosis(result_functions['imf0'][axes[ax]]))
        features.append(stats.kurtosis(result_functions['imf1'][axes[ax]]))
        features.append(stats.kurtosis(result_functions['imf2'][axes[ax]]))
    for ax in range(0, len(axes)):
        features.append(stats.kurtosis(result_functions['res0'][axes[ax]]))
        features.append(stats.kurtosis(result_functions['res1'][axes[ax]]))
        features.append(stats.kurtosis(result_functions['res2'][axes[ax]]))

    # conditionally provide graphs
    graphs = []
    if (draw_graphs):
        graphs.append({
            'name': 'IMF0',
            'X': axes_to_list(result_functions['imf0']),
            'y': np.linspace(0.0, raw_data.shape[0] * (1 / sampling_freq) * 1000, raw_data.shape[0] + 1).tolist(),
            'suggestedYMin': min(flatten(result_functions['imf0'].values())),
            'suggestedYMax': max(flatten(result_functions['imf0'].values())),
        })
        graphs.append({
            'name': 'RES0',
            'X': axes_to_list(result_functions['res0']),
            'y': np.linspace(0.0, raw_data.shape[0] * (1 / sampling_freq) * 1000, raw_data.shape[0] + 1).tolist(),
            'suggestedYMin': min(flatten(result_functions['res0'].values())),
            'suggestedYMax': max(flatten(result_functions['res0'].values()))
        })
        graphs.append({
            'name': 'IMF1',
            'X': axes_to_list(result_functions['imf1']),
            'y': np.linspace(0.0, raw_data.shape[0] * (1 / sampling_freq) * 1000, raw_data.shape[0] + 1).tolist(),
            'suggestedYMin': min(flatten(result_functions['imf1'].values())),
            'suggestedYMax': max(flatten(result_functions['imf1'].values()))
        })
        graphs.append({
            'name': 'RES1',
            'X': axes_to_list(result_functions['res1']),
            'y': np.linspace(0.0, raw_data.shape[0] * (1 / sampling_freq) * 1000, raw_data.shape[0] + 1).tolist(),
            'suggestedYMin': min(flatten(result_functions['res1'].values())),
            'suggestedYMax': max(flatten(result_functions['res1'].values()))
        })
        graphs.append({
            'name': 'IMF2',
            'X': axes_to_list(result_functions['imf2']),
            'y': np.linspace(0.0, raw_data.shape[0] * (1 / sampling_freq) * 1000, raw_data.shape[0] + 1).tolist(),
            'suggestedYMin': min(flatten(result_functions['imf2'].values())),
            'suggestedYMax': max(flatten(result_functions['imf2'].values()))
        })
        graphs.append({
            'name': 'RES2',
            'X': axes_to_list(result_functions['res2']),
            'y': np.linspace(0.0, raw_data.shape[0] * (1 / sampling_freq) * 1000, raw_data.shape[0] + 1).tolist(),
            'suggestedYMin': min(flatten(result_functions['res2'].values())),
            'suggestedYMax': max(flatten(result_functions['res2'].values()))
        })
    return {
            'features': features,
            'graphs': graphs
            }
