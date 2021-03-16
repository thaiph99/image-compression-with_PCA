import numpy as np
import json

dic = {'explained_variance_ratio_': [3.59421182e-01, 1.67083389e-01, 9.51737192e-02, 6.11929424e-02], 'noise_variance_': 101.60621838905892, 'n_features_in_': 1280,
       'batch_size_': 6400, 'n_components_': 100}

json_obj = json.dumps(dic)
print(json_obj)
