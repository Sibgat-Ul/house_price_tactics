import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import numpy as np
from collections import defaultdict
import tensorflow as tf;
from sklearn.preprocessing import StandardScaler
from scipy import stats

plt.style.use(style='default');
plt.rcParams['font.size'] = 12;
plt.figure(figsize=(9, 8));

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
train_data_quan = train_data[cluster1 + cluster3 + cluster6]
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
    train_cluster_4[cluster4[i]] = train_cluster_4[cluster4[i]].map(cluster_else)
    train_cluster_5[cluster5[i]] = train_cluster_5[cluster5[i]].map(cluster_else)
    train_cluster_7[cluster7[i]] = train_cluster_7[cluster7[i]].map(cluster_else)