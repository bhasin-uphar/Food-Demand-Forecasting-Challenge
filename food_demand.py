from keras.callbacks import ModelCheckpoint
from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten
import xgboost as xgb
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier

from sklearn.metrics import mean_squared_error
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

data = pd.read_csv('datasets/Food_Demand/train.csv')
meal = pd.read_csv('datasets/Food_Demand/meal.csv')
fulfilment = pd.read_csv('datasets/Food_Demand/fulfilment.csv')
test_data = pd.read_csv('datasets/Food_Demand/test.csv')

result = pd.merge(data,
                 meal,
                 on='meal_id', 
                 how='left')
result = pd.merge(result,
                 fulfilment,
                 on='center_id', 
                 how='left')

test_result = pd.merge(test_data,
                 meal,
                 on='meal_id', 
                 how='left')
test_result = pd.merge(test_result,
                 fulfilment,
                 on='center_id', 
                 how='left')

result.center_type = result.center_type.replace(['TYPE_C', 'TYPE_B', 'TYPE_A'],[1,2,3])
test_result.center_type = test_result.center_type.replace(['TYPE_C', 'TYPE_B', 'TYPE_A'],[1,2,3])
result.category = result.category.replace(['Beverages', 'Rice Bowl', 'Starters', 'Pasta', 'Sandwich',
       'Biryani', 'Extras', 'Pizza', 'Seafood', 'Other Snacks', 'Desert',
       'Soup', 'Salad', 'Fish'], [1,2,3,4,5,6,7,8,9,10,11,12,13,14])
test_result.category = test_result.category.replace(['Beverages', 'Rice Bowl', 'Starters', 'Pasta', 'Sandwich',
       'Biryani', 'Extras', 'Pizza', 'Seafood', 'Other Snacks', 'Desert',
       'Soup', 'Salad', 'Fish'], [1,2,3,4,5,6,7,8,9,10,11,12,13,14])
result.cuisine = result.cuisine.replace(['Thai', 'Indian', 'Italian', 'Continental'],[1,2,3,4])
test_result.cuisine = test_result.cuisine.replace(['Thai', 'Indian', 'Italian', 'Continental'],[1,2,3,4])

excel = pd.DataFrame()
excel['id'] = test_result['id']

result=result.drop(['center_id', 'meal_id', 'id'],axis=1)
test_result=test_result.drop(['center_id', 'meal_id', 'id'],axis=1)


clf2 = xgb.XGBRegressor(objective ='reg:linear', colsample_bytree = 0.3, learning_rate = 0.7,
                max_depth = 8, alpha = 20, n_estimators = 10, verbose=False)

clf1 = RandomForestRegressor(n_estimators=50, criterion='mse', max_depth=None, min_samples_split=3, min_samples_leaf=15,
                             min_weight_fraction_leaf=0.0, max_features='auto', max_leaf_nodes=None, min_impurity_decrease=0.0,
                             min_impurity_split=None, bootstrap=False, oob_score=False, n_jobs=5, random_state=None, verbose=5, warm_start=False)

clf3 = RandomForestClassifier(n_estimators='warn', criterion='gini', max_depth=None, min_samples_split=2, min_samples_leaf=2,
                              min_weight_fraction_leaf=0.0, max_features='auto', max_leaf_nodes=None, min_impurity_decrease=0.0,
                              min_impurity_split=None, bootstrap=False, oob_score=False, n_jobs=None, random_state=None, verbose=0,
                              warm_start=False, class_weight=None)
y = result['num_orders']
X = result.drop(['num_orders'],axis=1)
##data_dmatrix = xgb.DMatrix(data=X,label=y)

##X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=123)

clf=clf1
 
clf.fit(X,y)

preds = clf.predict(test_result)

excel['num_orders'] = preds.astype('int')
excel.to_csv("Food_Demand(clean)1.csv", index=False)



# My Rank		440		Score	62.3436087225


