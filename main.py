import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import numpy as np
from collections import defaultdict
import tensorflow as tf;
from sklearn.preprocessing import StandardScaler
from scipy import stats
from sklearn.decomposition import PCA
from sklearn.model_selection import GridSearchCV
from sklearn import linear_model
from sklearn.pipeline import  Pipeline, make_pipeline
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from pandas.api.types import is_string_dtype
from pandas.api.types import is_numeric_dtype
from sklearn.compose import make_column_selector as selector
from sklearn.model_selection import cross_validate
from sklearn.model_selection import KFold

def split_dataset(dataset, train_ratio=0.8):
    test_ration = 1 - train_ratio
    train_size = int(len(dataset) * train_ratio)
    train_set = dataset[:train_size]
    test_set = dataset[train_size:]
    return train_set, test_set

def label_encode(data, unique_val):
    encoding = {key: value for value, key in enumerate(unique_val)}
    data = data.map(encoding)
    print(encoding)
    return data

def applyChange(data):
    # NA does not mean missing values, it means that the house does not have that feature
    # Fill NA with 0 for numbers/rank and None for objects/types
    data['MSSubClass'] = data['MSSubClass'].astype(object)
    data['LotArea'] = data['LotArea'].fillna(0)
    data['LotFrontage'] = data['LotFrontage'].fillna(0)
    data['Alley'] = data['Alley'].fillna('None')

    data["MasVnrType"] = data["MasVnrType"].fillna("None")
    data["MasVnrArea"] = data["MasVnrArea"].fillna(0)

    data['BsmtQual'] = data['BsmtQual'].fillna('None')
    data['BsmtCond'] = data['BsmtCond'].fillna('None')
    data['BsmtExposure'] = data['BsmtExposure'].fillna('None')
    data['BsmtFinType1'] = data['BsmtFinType1'].fillna('None')
    data['BsmtFinType2'] = data['BsmtFinType2'].fillna('None')

    data['BsmtFinSF1'] = data['BsmtFinSF1'].fillna(0)
    data['BsmtFinSF2'] = data['BsmtFinSF2'].fillna(0)

    data['FireplaceQu'] = data['FireplaceQu'].fillna('None')
    data['Electrical'] = data['Electrical'].fillna('Mix')

    data['GarageType'] = data['GarageType'].fillna('None')
    data['GarageYrBlt'] = data['GarageYrBlt'].fillna(min(data['GarageYrBlt']))
    data['GarageFinish'] = data['GarageFinish'].fillna('None')
    data['GarageQual'] = data['GarageQual'].fillna('None')
    data['GarageCond'] = data['GarageCond'].fillna('None')

    data['PoolQC'] = data['PoolQC'].fillna('None')
    data['Fence'] = data['Fence'].fillna('None')
    data['MiscFeature'] = data['MiscFeature'].fillna('None')


    data['1stFlrSF'] = data['1stFlrSF'].fillna(data['1stFlrSF'].median())
    data['TotalBsmtSF'] = data['TotalBsmtSF'].fillna(data['TotalBsmtSF'].median())
    data['GrLivArea'] = data['GrLivArea'].fillna(data['GrLivArea'].median())

    return data;

def normalize(train, test):
    mean = train.mean(numeric_only=True)
    std = test.std(numeric_only=True)

    print(mean, std)

def applyDimReduction(data):
    data = data.drop(['BsmtCond', 'GarageCond'], axis=1)
    #print(data.head(5))
    #data[['LotFrontage', 'LotArea']] = data[['LotFrontage', ['LotArea']]].apply(lambda x: (x - x.min()) / (x.max() - x.min()))
    data[['LotFrontage', 'LotArea']] = StandardScaler().fit_transform(data[['LotFrontage', 'LotArea']])
    data['LotPCA'] =  PCA(n_components=1).fit_transform(data[['LotFrontage', 'LotArea']])

    data[['1stFlrSF', 'TotalBsmtSF', 'GrLivArea']] = StandardScaler().fit_transform(data[['1stFlrSF', 'TotalBsmtSF', 'GrLivArea']])
    data['AreaPCA'] = PCA(n_components=1).fit_transform(data[['1stFlrSF', 'TotalBsmtSF', 'GrLivArea']])

    data = data.drop(['LotFrontage', 'LotArea', '1stFlrSF', 'TotalBsmtSF', 'GrLivArea'], axis=1)
    #print(data.head(5))
    return data

def applyDimReduction2(data):
    data = data.drop(['GarageCars', 'GarageYrBlt', 'BedroomAbvGr'], axis=1)
    return data

def get_score(model, cv, random_state):
    "Scores (mean_squared_error) for given data, model with cross-validation."

    # Perform cross-validation using the specified model and data
    CV = cross_validate(model, x_train, y_train, scoring='neg_mean_squared_error',
                        cv=KFold(n_splits=cv, shuffle=True, random_state=random_state),
                        return_estimator=True)

    # Calculate the number of features used by each estimator (model) during cross-validation
    try:
        n_features = np.min([e[:-1].get_feature_names_out().size for e in CV['estimator']])
    except:
        # If the number of features cannot be determined, use the total number of features in the training data
        n_features = x_train.shape[1]

    # Calculate the mean and standard deviation of the RMSE scores from cross-validation
    scores = [n_features, np.mean(np.sqrt(-CV['test_score'])), np.std(np.sqrt(-CV['test_score']))]

    return scores

def observe():

    train_data = pd.read_csv('./home_data/train.csv')
    test_data = pd.read_csv('./home_data/test.csv')

    train_data = applyChange(train_data)

    # print()
    # described_data = train_data.describe().transpose()
    # described_data.to_csv('./home_data/described_data.csv')

    trainDataSub1 = train_data[
        ['LotFrontage','MasVnrArea','BsmtFinSF1','1stFlrSF',
         'TotalBsmtSF','GrLivArea','GarageArea','OpenPorchSF',
         'TotRmsAbvGrd','WoodDeckSF','BedroomAbvGr','SalePrice']
    ]

    # plt.rcParams['figure.facecolor'] = 'white'
    # train_data.hist(bins=50, figsize=(20,15))
    #
    # plt.rcParams['figure.facecolor'] = 'white'
    # pd.plotting.scatter_matrix(trainDataSub1, alpha=0.2, figsize=(20, 20), diagonal='kde');
    #
    # plt.show()

    cluster1 = ['LotFrontage','LotArea'] # yes strong correlation
    cluster2 = ['Street','Alley'] # categorical
    cluster3 = ['OverallQual','OverallCond']
    cluster4 = ['ExterQual','ExterCond'] # categorical
    cluster5 = ['BsmtQual','BsmtCond'] # categorical
    cluster6 = ['1stFlrSF','TotalBsmtSF','GrLivArea'] # yes strong correlation
    cluster7 = ['GarageQual','GarageCond'] # categorical

    # Numeric cols
    train_data_num = train_data[cluster1 + cluster3 + cluster6]
    # plt.rcParams['figure.facecolor'] = 'white'
    # pd.plotting.scatter_matrix(train_data_quan, alpha=0.2, figsize=(20, 20), diagonal='kde');
    # plt.show()

    # Categorical
    cluster2_map = {'Grvl': 2, 'Pave': 1, 'None': 0};
    cluster_else = {'Ex': 5, 'Gd': 4, 'TA': 3, 'Fa': 2, 'Po': 1, 'None': 0};

    train_cluster_2 = train_data[cluster2].copy()
    train_cluster_4 = train_data[cluster4].copy()
    train_cluster_5 = train_data[cluster5].copy()
    train_cluster_7 = train_data[cluster7].copy()

    for i in range(len(cluster2)):
        train_cluster_2[cluster2[i]] = train_cluster_2[cluster2[i]].map(cluster2_map)

    for i in range(len(cluster4)):
        train_cluster_4[cluster4[i]] = train_cluster_4[cluster4[i]].map(cluster_else)

    for i in range(len(cluster5)):
        train_cluster_5[cluster5[i]] = train_cluster_5[cluster5[i]].map(cluster_else)

    for i in range(len(cluster7)):
        train_cluster_7[cluster7[i]] = train_cluster_7[cluster7[i]].map(cluster_else)

    t1max = train_data[cluster1[0]].quantile(0.99);
    t1min = train_data[cluster1[0]].quantile(0.01);
    t2max = train_data[cluster1[1]].quantile(0.99);
    t2min = train_data[cluster1[1]].quantile(0.01);

    print(t1max, t1min, t2max, t2min);

    train_data = train_data.dropna()
    train_data = train_data.drop(['Id'], axis=1)

    train_data = applyDimReduction(train_data)
    #dropping values with 0.7+ corr
    corr_matrix = train_data.corr().abs()
    s = corr_matrix.unstack()
    sort_corr = s.sort_values(kind="quicksort", ascending=False).drop_duplicates()
    print(sort_corr[sort_corr < 1.0][:30])

    train_data = applyDimReduction2(train_data)
    print(len(train_data.columns))
    corr_matrix = train_data.corr().abs()
    s = corr_matrix.unstack()
    sort_corr = s.sort_values(kind="quicksort", ascending=False).drop_duplicates()
    print(sort_corr[sort_corr < 1.0][:30])


    train_data = pd.get_dummies(train_data, drop_first=True)

#A. Lasso model with data as it is (no ordinal transformation)
train_data = pd.read_csv('./home_data/train.csv')
test_data = pd.read_csv('./home_data/test.csv')
x_train = train_data.drop(['SalePrice'], axis=1)
y_train = np.log1p(train_data['SalePrice'])

indexes = x_train[(train_data['GrLivArea'] < 4000) & (train_data['SalePrice']<3000)].index
x_train = x_train.drop(indexes)
y_train = y_train.drop(indexes)

var_num = []
var_cat = []

for col in train_data:
    if is_numeric_dtype(train_data[col]):
        var_num.append(col)
    else:
        var_cat.append(col)

rem_num = ['SalePrice', 'Id', 'LotFrontage', 'LotArea','1stFlrSF','TotalBsmtSF','GrLivArea','GarageCars','GarageYrBlt','BedroomAbvGr']
rem_cat = ['BsmtCond','GarageCond']

var_num = [x for x in var_num if x not in rem_num]
var_num.append(['LotPCA'])
var_num.append(['AreaPCA'])
var_cat = [x for x in var_cat if x not in rem_cat]

class CommonPreprocess(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass
    def fit(self, X, y=None):
        return self
    def transform(self, X, y=None):
        X = applyChange(X)
        X = applyDimReduction(X)
        X = applyDimReduction2(X)
        return X

class MyImputer(BaseEstimator, TransformerMixin):
    def __init__(self, strategy):
        self.strategy = strategy
    def fit(self, X, y=None):
        return self
    def transform(self, X, y=None):
        if self.strategy == 'median':
            X = X.fillna(X.median())
        elif self.strategy == 'most_frequent':
            X = X.fillna(X.mode().iloc[0])
        return X

class MyLog(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass
    def fit(self, X, y=None):
        return self
    def transform(self, X, y=None):
        for xcol in X.columns:
            for x in X[xcol]:
                if x+1 > 0:
                    X[xcol] = np.log(x + 1)
                else:
                    X[xcol] = x + 1
        return X

cat_pipe = Pipeline([
    ('imputer', MyImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(categories='auto' ,handle_unknown='ignore')),
])

num_pipe = Pipeline([
    ('imputer', MyImputer(strategy='median')),
    ('scaler', StandardScaler()),
])

init_pipe = ColumnTransformer([
    ('num_pipe', num_pipe, selector(dtype_exclude=['object'])),
    ('cat_pipe', cat_pipe, selector(dtype_include=['object'])),
], remainder='drop')

model_1 = Pipeline([
    ('preprocess', CommonPreprocess()),
    ('init_pipe', init_pipe),
    ('model_1', linear_model.Lasso(alpha=0.0005, random_state=1)),
])

param = {'model_1__alpha': [0.0003, 0.0004, 0.0005, 0.0006, 0.0007]} # this was found heuristically

search = GridSearchCV(model_1, param, cv=5, scoring='neg_mean_squared_error', verbose=5)
result_1 = search.fit(x_train, y_train)
print(result_1.best_params_)
print(result_1.best_score_)

results = pd.DataFrame([],columns=['no_feature', 'mean_cv_error','std_cv_error'])
results.loc["Lasso(0.0005) all dummy"] = get_score(result_1.best_estimator_, random_state=15, cv=5)

model_2 = Pipeline([
    ('preprocess', CommonPreprocess()),
    ('init_pipe', init_pipe),
    ('model_2', linear_model.Ridge()),
])

param = {'model_2__alpha': [0.7, 0.8, 0.9]} # this was found heuristically

search = GridSearchCV(model_2, param, cv=5, scoring='neg_mean_squared_error', verbose=3)
result_2 = search.fit(x_train, y_train)
print(f'Best Score: {result_2.best_score_}' % result_2.best_score_)
print(f'Best Estimator: {result_2.best_estimator_}')
results.loc["Ridge(0.9) all dummy"] = get_score(result_2.best_estimator_, random_state=15, cv=5)

from sklearn.preprocessing import OrdinalEncoder

ordMap = { # set to be the higher the better
    'LotShape': {'Reg': 3, 'IR1': 2, 'IR2': 1, 'IR3': 0},
    'Utilities': {'AllPub': 3, 'NoSewr': 2, 'NoSeWa': 1, 'ELO': 0},
    'LandSlope': {'Gtl': 2, 'Mod': 1, 'Sev': 0},
    'ExterQual': {'Ex': 4, 'Gd': 3, 'TA': 2, 'Fa': 1, 'Po': 0},
    'ExterCond': {'Ex': 4, 'Gd': 3, 'TA': 2, 'Fa': 1, 'Po': 0},
    'BsmtQual': {'Ex': 5, 'Gd': 4, 'TA': 3, 'Fa': 2, 'Po': 1, 'None': 0},
    'BsmtExposure': {'Gd': 4, 'Av': 3, 'Mn': 2, 'No': 1, 'None': 0},
    'HeatingQC': {'Ex': 4, 'Gd': 3, 'TA': 2, 'Fa': 1, 'Po': 0},
    'KitchenQual': {'Ex': 4, 'Gd': 3, 'TA': 2, 'Fa': 1, 'Po': 0},
    'Functional': {'Typ': 7, 'Min1': 6, 'Min2': 5, 'Mod': 4, 'Maj1': 3, 'Maj2': 2, 'Sev': 1, 'Sal': 0},
    'FireplaceQu': {'Ex': 5, 'Gd': 4, 'TA': 3, 'Fa': 2, 'Po': 1, 'None': 0},
    'GarageQual': {'Ex': 5, 'Gd': 4, 'TA': 3, 'Fa': 2, 'Po': 1, 'None': 0},
    'PavedDrive': {'Y': 2, 'P': 1, 'N': 0},
    'Fence': {'GdPrv': 4, 'MnPrv': 3, 'GdWo': 2, 'MnWw': 1, 'None': 0},
}

class ToOrdinal(BaseEstimator,TransformerMixin): # due to complexities with OrdinalEncoder
    def __init__(self, variables=None):
        self.variables = variables

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        # if self.variable in ordMap:
        #     X[self.variable]  = X[self.variable].map(ordMap[self.variable])
        for variable in self.variables:
            X[variable] = X[variable].map(ordMap[variable])
            # X = X.drop(variable, axis=1)
        return X

ordCats = ['LotShape','Utilities','LandSlope','ExterQual','ExterCond','BsmtQual','BsmtExposure','HeatingQC','KitchenQual','Functional','FireplaceQu','GarageQual','PavedDrive','Fence']

# to dummy for rest of categorical
cat_dummy_pipe = ColumnTransformer([
    ('cat_dummy', OneHotEncoder(categories='auto', handle_unknown='ignore',), selector(dtype_include=['object'])),
], remainder='passthrough')

# change categorical to ordinal
cat_pipe = Pipeline([
    ('imputer_cat', MyImputer(strategy='most_frequent')),
    ('ordinal', ToOrdinal(variables=ordCats)),
    ('cat_dummy_pipe', cat_dummy_pipe)
])

num_pipe = Pipeline([
    ('imputer_num', MyImputer(strategy='median')), # for numeric
    # ('log_transform', MyLog()),
    ('scaler', StandardScaler()) # this would also scale the ToOrdinal() output? probably not
])

initial_pipe3 = ColumnTransformer([
    ('cat_pipe', cat_pipe, selector(dtype_include=['object'])), # for categorical
    ('num_pipe', num_pipe, selector(dtype_exclude=['object'])), # for numeric
], remainder='drop')

modelV3 = Pipeline([
    ('common_preprocess', CommonPreprocess()),
    ('initialize', initial_pipe3),
    ('modelV3', linear_model.Lasso())
])

param = {'modelV3__alpha': [0.0003, 0.0004, 0.0005, 0.0006]}

search = GridSearchCV(modelV3, param, cv=5, scoring='neg_mean_squared_error', verbose=5)
resultV3 = search.fit(x_train, y_train)
# Lasso with ordinality (no scaling - uniform distance)
print(f'Best Score: {resultV3.best_score_}' % resultV3.best_score_)
print(f'Best Estimator: {resultV3.best_estimator_}') # the results may be as such because there are too many variables

results.loc["Lasso(0.0005) ordinal dummy"] = get_score(resultV3.best_estimator_, random_state=15, cv=5)

from sklearn.ensemble import VotingRegressor
from sklearn.model_selection import cross_val_score
# scoresModelV1 = cross_val_score(modelV1, xTrain, yTrain, cv=5, scoring='neg_mean_squared_error')
# scoresModelV3 = cross_val_score(modelV3, xTrain, yTrain, cv=5, scoring='neg_mean_squared_error')
ensemble = VotingRegressor(estimators=[('dummy', model_1), ('ord1', modelV3)], weights=[1, 1])
results.loc["Ensemble top 2"] = get_score(ensemble, random_state=15, cv=5)
scoresEnsemble = cross_val_score(ensemble, x_train, y_train, cv=5, scoring='neg_mean_squared_error')

print(np.sqrt(-scoresEnsemble).mean())
ensembleModel = ensemble.fit(x_train,y_train)
sampleSubmit = pd.read_csv("./home_data/sample_submission.csv")
sampleSubmit['SalePrice'] = np.exp(ensemble.predict(test_data))
sampleSubmit.to_csv("submission.csv", index=False)