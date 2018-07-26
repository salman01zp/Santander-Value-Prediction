import  numpy as np 
import pandas as pd 
from sklearn.model_selection import train_test_split
from scipy.stats import skew, kurtosis

import lightgbm as lgb



def get_data():
	print('Reading data')
	train_data = pd.read_csv('../input/train.csv', nrows=None)
	test_data = pd.read_csv('..input/test.csv', nrows=None)
	print('Train shape', train_data.shape, 'Test Shape', test_data.shape)
	return train_data, test_data

def add_new_features(train_data, test_data):
	train_data.replace(0, np.nan, inplace = True)
	test_data.replace(0, np.nan , inplace= True)

	original_features = [f for f in data.columns if f not in ['target', 'ID']]
	for df in [train_data, test_data]:
		df['nans'] = df[original_features].isnull().sum(axis=1)
		df['median'] = df['original_features'].median(axis=1)
		df['sum'] = df['original_features'].sum(axis=1)
		df['mean'] = df['original_features'].mean(axis=1)
		df['std'] = df['original_features'].std(axis=1)
		df['kur'] = df['original_features'].kurtosis(axis=1)

		return train_data, test_data
def get_selected_features():
	return[
		'f190486d6', 'c47340d97', 'eeb9cd3aa', '66ace2992', 'e176a204a',
        '491b9ee45', '1db387535', 'c5a231d81', '0572565c2', '024c577b9',
        '15ace8c9f', '23310aa6f', '9fd594eec', '58e2e02e6', '91f701ba2',
        'adb64ff71', '2ec5b290f', '703885424', '26fc93eb7', '6619d81fc',
        '0ff32eb98', '70feb1494', '58e056e12', '1931ccfdd', '1702b5bf0',
        '58232a6fb', '963a49cdc', 'fc99f9426', '241f0f867', '5c6487af1',
        '62e59a501', 'f74e8f13d', 'fb49e4212', '190db8488', '324921c7b',
        'b43a7cfd5', '9306da53f', 'd6bb78916', 'fb0f5dbfe', '6eef030c1'
    ]

	


def fit_predict(train_data, y , test_data)
	
	features = get_selected_features() + ['nans', 'median', 'mean','std','sum','kur']
	X_train , y_train, X_test, y_test = train_test_split(train_data[features], y, test_size=0.2, random_state=42)
		
	 params = {
        "objective" : "regression",
        "metric" : "rmse",
        "num_leaves" : 40,
        "learning_rate" : 0.005,
        "bagging_fraction" : 0.7,
        "feature_fraction" : 0.5,
        "bagging_frequency" : 5,
        "bagging_seed" : 42,
        "verbosity" : -1,
        "seed": 42
    }

    lgbtrain = lgb.Dataset(X_train, label= y_train)
    lgbval = lgb.Dataset(X_test, label = y_test)
    evels_result = {}
    model = lgb.train(params, lgbtrain, 5000, valid_sets=[lgbval],
    					early_stopping_rounds=100,
    					verbose_eval=50,
    					evals_result = evals_result)

    pred_test = np.exmp1(model.predict(test_data, num_iteration= model.best_iteration))
    print("LightGBM Training Completed...")
    return pred_test, model ,evals_result


def main():
	train_data, test_data = get_data()

	y = train_data[['ID', 'target']].copy()
	del train_data['target'], train_data['ID']
	sub = test_data[['ID']].copy()
	del test_data['ID']


	gc.collect()

	train_data, test_data = add_new_features(train_data, test_data)

	pred_test , model, evals_result= fit_predict(train_data, y , test_data)

	sub = pd.read_csv('../input/sample_submission.csv')
	sub['target']= pred_test
	print(sub.head())
	sub.to_csv('final_result.csv', index=False)
