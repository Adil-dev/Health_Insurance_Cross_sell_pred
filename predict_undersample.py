import pandas as pd 
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
import lightgbm as lgb
import pickle
from imblearn.under_sampling import RandomUnderSampler 
import os

DATADIR = 'deptest'

filename = 'train.csv'

stock_df = pd.read_csv(os.path.join(DATADIR, filename), sep=',', index_col=['id'])

va = {'> 2 Years': 2, '1-2 Year': 1, '< 1 Year': 0}
gen = {'Male' : 0, 'Female' : 1}
vg = {'Yes' : 1, 'No' : 0}
stock_df['Vehicle_Age'] = stock_df['Vehicle_Age'].map(va)
stock_df['Gender'] = stock_df['Gender'].map(gen)
stock_df['Vehicle_Damage'] = stock_df['Vehicle_Damage'].map(vg)

num_feat = ['Age', 'Vintage']

cat_feat = [
    'Gender', 'Previously_Insured', 'Vehicle_Age', 'Vehicle_Damage',
    'Driving_License', 'Policy_Sales_Channel', 'Region_Code'
]

dead_outliers_df = stock_df.query('Annual_Premium <= 100000')

num_feat = ['Age', 'Vintage', 'Annual_Premium']

cat_feat = [
    'Gender', 'Previously_Insured', 'Vehicle_Age', 'Vehicle_Damage',
    'Driving_License', 'Policy_Sales_Channel', 'Region_Code'
]

scl = StandardScaler()

num_scl = pd.DataFrame(scl.fit_transform(dead_outliers_df[num_feat]))
num_scl.index = dead_outliers_df[num_feat].index
num_scl.columns = dead_outliers_df[num_feat].columns
X_ = pd.concat([num_scl, dead_outliers_df[cat_feat]], axis=1)

X = X_
y = dead_outliers_df.Response

undersamp = RandomUnderSampler(sampling_strategy=0.9)
X_over, y_over = undersamp.fit_resample(X, y)

X_train, X_valid, y_train, y_valid = train_test_split(X_over, y_over, train_size=0.8, shuffle=True, random_state=34)

params = {
    'reg_lambda': 1.8,
    'reg_alpha': 0.9,
    'num_leaves': 80,
    'min_child_weight': 1,
    'max_depth': 6,
    'learning_rate': 0.12,
    'feature_fraction': 0.5,
    'bagging_fraction': 0.5,
    'objective': 'binary',
    "boosting_type": "gbdt",
    "bagging_seed": 23,
    "metric": 'auc',
    "verbosity": -1,
    'num_iterations' : 90
}

clf = lgb.LGBMClassifier(**params)

clf.fit(X_train, y_train)

pickle_out_under = open('clf_unuder.pkl', 'wb')
pickle.dump(clf, pickle_out_under)
pickle_out_under.close()
