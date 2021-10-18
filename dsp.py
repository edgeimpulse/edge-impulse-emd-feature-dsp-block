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

def generate_features(draw_graphs, raw_data, axes, sampling_freq, hht_len, **kwargs):
    # workaround for weird issue where hht_len is str
    hht_len = int(hht_len)
    # features is a 1D array, reshape so we have a matrix with one raw per axis
    raw_data = raw_data.reshape(int(len(raw_data) / len(axes)), len(axes))

    # features is a fixed size 48x1 dimension list
    features = []
    graphs = []

    graph_data = {'imf0' : {}, 'imf1' : {}, 'imf2' : {}, 'res0' : {}, 'res1' : {}, 'res2' : {}}

    # split out the data from all axes
    for ax in range(0, len(axes)):
        X = []
        for ix in range(0, raw_data.shape[0]):
            X.append(raw_data[ix][ax])

        # X now contains only the current axis
        fx = np.array(X)

        imf = []
        res = []

        # get first three imfs and residuals
        imf.append(get_next_imf(fx))
        res.append(fx - imf[0])
        imf.append(get_next_imf(res[0]))
        res.append(res[0] - imf[1])
        imf.append(get_next_imf(fx - imf[0] - imf[1]))
        res.append(res[1] - imf[2])

        # compute statistical features for imfs and residuals
        for i in imf:
            features.append(np.mean(i))
            features.append(np.std(i))
            features.append(stats.skew(i))
            features.append(stats.kurtosis(i))
        
        # compute statistical features for imfs and residuals
        for i in res:
            features.append(np.mean(i))
            features.append(np.std(i))
            features.append(stats.skew(i))
            features.append(stats.kurtosis(i))

        # draw graphs conditional in-loop
        if (draw_graphs):
            graph_data['imf0'][axes[ax]] = imf[0].tolist()
            graph_data['res0'][axes[ax]] = res[0].tolist()
            graph_data['imf1'][axes[ax]] = imf[1].tolist()
            graph_data['res1'][axes[ax]] = res[1].tolist()
            graph_data['imf2'][axes[ax]] = imf[2].tolist()
            graph_data['res2'][axes[ax]] = res[2].tolist()

    if (draw_graphs):
        graphs.append({
            'name': 'IMF0',
            'X': graph_data['imf0'],
            'y': np.linspace(0.0, raw_data.shape[0] * (1 / sampling_freq) * 1000, raw_data.shape[0] + 1).tolist(),
            'suggestedYMin': min(flatten(graph_data['imf0'].values())),
            'suggestedYMax': max(flatten(graph_data['imf0'].values())),
        })
        graphs.append({
            'name': 'RES0',
            'X': graph_data['res0'],
            'y': np.linspace(0.0, raw_data.shape[0] * (1 / sampling_freq) * 1000, raw_data.shape[0] + 1).tolist(),
            'suggestedYMin': min(flatten(graph_data['res0'].values())),
            'suggestedYMax': max(flatten(graph_data['res0'].values()))
        })
        graphs.append({
            'name': 'IMF1',
            'X': graph_data['imf1'],
            'y': np.linspace(0.0, raw_data.shape[0] * (1 / sampling_freq) * 1000, raw_data.shape[0] + 1).tolist(),
            'suggestedYMin': min(flatten(graph_data['imf1'].values())),
            'suggestedYMax': max(flatten(graph_data['imf1'].values()))
        })
        graphs.append({
            'name': 'RES1',
            'X': graph_data['res1'],
            'y': np.linspace(0.0, raw_data.shape[0] * (1 / sampling_freq) * 1000, raw_data.shape[0] + 1).tolist(),
            'suggestedYMin': min(flatten(graph_data['res1'].values())),
            'suggestedYMax': max(flatten(graph_data['res1'].values()))
        })
        graphs.append({
            'name': 'IMF2',
            'X': graph_data['imf2'],
            'y': np.linspace(0.0, raw_data.shape[0] * (1 / sampling_freq) * 1000, raw_data.shape[0] + 1).tolist(),
            'suggestedYMin': min(flatten(graph_data['imf2'].values())),
            'suggestedYMax': max(flatten(graph_data['imf2'].values()))
        })
        graphs.append({
            'name': 'RES2',
            'X': graph_data['res2'],
            'y': np.linspace(0.0, raw_data.shape[0] * (1 / sampling_freq) * 1000, raw_data.shape[0] + 1).tolist(),
            'suggestedYMin': min(flatten(graph_data['res2'].values())),
            'suggestedYMax': max(flatten(graph_data['res2'].values()))
        })
    return {
            'features': features,
            'graphs': graphs
            }
