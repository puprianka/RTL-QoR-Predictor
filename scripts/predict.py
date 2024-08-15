import os
import sys
import pickle
import numpy as np
from math import sqrt

def main():
    
    # load numpy features
    np_file = sys.argv[1]
    print(np_file)
    feat_vec = np.load(np_file).flatten('F')
    
    # Append synth params
    synth_params = {'AREA': 10, 'GPL_TIMING_DRIVEN': 1, 'PLACE_DENSITY': 0.85, 'CORE_ASPECT_RATIO': 1.0, 'clock_period': 2.5, 'delay_pct': 0.2, 'uncertainty': 0.025}
    synth_param_vals = list(synth_params.values())
    
    # load and run pickle models
    pkl_file = sys.argv[2]
    print(pkl_file)
    slack_pickle_model = pickle.load(open(pkl_file, "rb"))
    
    slack_pred = slack_pickle_model.predict(np.append(feat_vec, synth_param_vals).reshape(1, -1))

    if "model_7" in pkl_file:
        print(f"Predicted TNS: -{sqrt(abs(slack_pred[0]))/1000: 0.4f} ns")
    else:
        print(f"Predicted TNS: -{sqrt(abs(slack_pred[0])): 0.4f} ns")
    
    
if __name__ == '__main__':
    main()
