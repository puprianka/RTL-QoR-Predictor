# Import things
import pandas as pd
import xgboost as xgb
from random import gauss
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error
import argparse as ap
from pathlib import Path
import csv
import numpy as np
import altair as alt
from scipy.stats import pearsonr
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import pickle

alt.data_transformers.disable_max_rows()

def custom_obj(y_pred, y_true):
    gradient = np.where(y_pred < 0, -2 * y_true, 2 * (y_pred - y_true))
    hessian = np.where(y_pred < 0, 2, 2)
    return gradient, hessian

def main():
    parser = ap.ArgumentParser()
    parser.add_argument("-dfile", required=True, help="Provide the dyn_pwr datafile")
    parser.add_argument("-features", required=True, help="Provide the feature dir")
    parser.add_argument("-eval", required=False, help="Provide some eval design name")
    parser.add_argument("-scale", required=False, default=1, help="Provide scaling? Why")

    args = parser.parse_args()
    x_params = ['ABC_MODE', 'GPL_MODE', 'CORE_ASPECT_RATIO', 'SYNTH_HIERARCHICAL', 'CORE_UTILIZATION', 
                'clk_scaling', 'uncertainty', 'max_fanout', 'max_capacitance'] #, 'clock_period']

    # Load the data run data
    X_list=[]
    y_list=[]
    X_val_list = []
    y_val_list = []
    y_data = {}
    with open(args.dfile, 'r') as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            if row['design'] == '':
                continue
            if row['dyn_pwr'] == 'NA':
                continue
            # Only take rows wth "clk_scaling == 0.8, so we have much dyn_pwr"
            if float(row['clk_scaling']) != 0.8 or float(row['dyn_pwr']) == 0.0:
                continue
            if row['ABC_MODE'] == 'NA':
                continue
            if row['activity_factor'] != '0.1':
                continue
            if row['design'] not in y_data:
                y_data[row['design']] = [row]
            else:
                y_data[row['design']].append(row)
    print(f"Synth data form {len(y_data)}")

    # Load the ast data
    mdict = {}
    design_name = []
    for xnp in Path(args.features).iterdir():
        if "npy" in xnp.name:
            xid = xnp.stem
            if xid in y_data:
                print(f"Loading {xid}")
                xarr = np.load(xnp)
                for row in y_data[xid]:
                    # Check if there is NA for any of the params
                    param_vals = [row[k] for k in x_params]
                    if "NA" in param_vals or "NA" in row['dyn_pwr']:
                        continue

                    x_data = xarr.flatten('F')
                    mdict.setdefault(xid, True)
                    # x_data = xarr.flatten()[4:]
                    # x_data = []
                    for colname in x_params:
                        x_data = np.append(x_data, float(row[colname]))
                    if args.eval in xid:
                        # y_val_list.append(float(row['dyn_pwr'])*gauss(1, 0.02)/scale)
                        y_val_list.append(float(row['dyn_pwr']))
                        X_val_list.append(x_data.tolist())
                    else:
                        # y_list.append(float(row['dyn_pwr'])*gauss(1, 0.02)/scale)
                        y_list.append(float(row['dyn_pwr']))
                        design_name.append(xid)
                        X_list.append(x_data.tolist())
            else:
                print(f"Cant find run data for {xid}")
    
    print(f"Feature and Label merged for {len(mdict)} designs")
    
    print(f"Training with {len(y_list)}")
    print(f"Testing with {len(y_val_list)}")
    # exit()

    X_train = np.array(X_list)
    X_test = np.array(X_val_list)
    y_train = np.array(y_list)
    y_test = np.array(y_val_list)

    ## build a model
    # regr = xgb.XGBRegressor(n_estimators=30, max_depth=5, colsample_bytree=0.01, reg_alpha=0.2, reg_lambda=0.2)
    regr = RandomForestRegressor(n_estimators=100, max_depth=20)
    # regr = MLPRegressor(hidden_layer_sizes=(64, 16, 8), random_state=42, 
    #                     solver='adam', max_iter=600, activation='identity')
    # regr = KernelRidge(alpha=1.0)
    pipe = Pipeline([
        ('scaler', StandardScaler()),
        ('pca', PCA(n_components=0.95)),  # Adjust n_components as needed
        ('regr', regr)  # Replace with your desired model
    ])
    # pipe = make_pipeline(regr)

    print("Training Model")
    pipe.fit(X_train, y_train)
    print(f"Training Done")

    y_train_pred=pipe.predict(X_train)
    print("\nTraining Score:", (pipe.score(X_train, y_train)))
    print("Training Error: ", mean_squared_error(y_train, y_train_pred, squared=False))
    print("Correlation: ", pearsonr(y_train, y_train_pred).statistic)
    print(f"Training Range: {(max(y_train)-min(y_train))}")

    results = pd.DataFrame().from_dict({'design': design_name, 'y_true': y_train, 'y_pred': y_train_pred})
    # results.to_csv(f'pwr_train_pred_7.csv')
    results.to_csv(f'pwr_train_pred_45.csv')


    y_test_pred=pipe.predict(X_test)
    print("\nValidation Score:", (pipe.score(X_test, y_test)))
    print("Validation Error: ", mean_squared_error(y_test, y_test_pred, squared=False))
    print("Correlation: ", pearsonr(y_test, y_test_pred).statistic)
    print(f"Validation Range: {(max(y_test)-min(y_test))}")

    # plot_data = pd.DataFrame().from_dict({'y_true': y_test, 'y_pred': y_test_pred})
    # plot_data.to_csv(f'{args.eval}_pred_45.csv')

    # with open('dyn_pwr_model_nangate7.pkl','wb') as f:
    #     pickle.dump(regr,f)

    with open('dyn_pwr_model_nangate45.pkl','wb') as f:
        pickle.dump(regr,f)

    exit()

    scatter = alt.Chart(plot_data).mark_point().encode(
        x='y_true',
        y='y_pred',
        tooltip=['y_true', 'y_pred']
    ).interactive()

    scatter.show()

if __name__ == '__main__':
    main()