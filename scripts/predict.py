import sys
import pickle
import numpy as np

def main():
    
    # load numpy features
    np_file = sys.argv[1]
    print(np_file)
    feat_vec = np.load(np_file).flatten('F')
    
    # Append synth params
    synth_params = {'ABC_MODE': 1, 'GPL_MODE': 1, 'CORE_ASPECT_RATIO': 0.5, 'SYNTH_HIERARCHICAL': 0, 'CORE_UTILIZATION': 50, 
                'clk_scaling': 1.0, 'uncertainty': 0.05, 'max_fanout': 20, 'max_capacitance': 0.1}
    synth_param_vals = list(synth_params.values())
    
    # load and run pickle models
    pkl_file = sys.argv[2]
    print(pkl_file)
    pickle_model = pickle.load(open(pkl_file, "rb"))
    
    pred = pickle_model.predict(np.append(feat_vec, synth_param_vals).reshape(1, -1))

    if "slack" in pkl_file:
      result = f"Prediction: {-1*abs(pred[0])/1000: 0.4f}"
    elif "pwr" in pkl_file:
      result = f"Prediction: {abs(pred[0])/1000: 0.4f}"
    else:
      result = f"Prediction: {abs(pred[0])/1000: 0.4f}"    
    
    print(result)
    
if __name__ == '__main__':
    main()
