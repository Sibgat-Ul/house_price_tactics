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

df_train = train_data.copy()
df_test = test_data.copy()

salesPrice = df_train['SalePrice']
# print(salesPrice.describe())
# print(salesPrice.skew(), salesPrice.kurt())

var = 'GrLivArea'
data = pd.concat([df_train['SalePrice'], df_train[var]], axis=1)
data.plot.scatter(x=var, y='SalePrice', ylim=(0, 800000));
#plt.show()

var = 'TotalBsmtSF'
data = pd.concat([df_train['SalePrice'], df_train[var]], axis=1)
data.plot.scatter(x=var, y='SalePrice', ylim=(0, 800000));
#plt.show()

var = 'OverallQual'
data = pd.concat([df_train['SalePrice'], df_train[var]], axis=1)
f, ax = plt.subplots(figsize=(8, 6))
sns.boxplot(x=var, y="SalePrice", data=data);
#plt.show()

corrmat = df_train.corr(numeric_only=True);
f, ax = plt.subplots(figsize=(12, 9))
sns.heatmap(corrmat, square=True);
#plt.show()

cols = corrmat.nlargest(10, 'SalePrice')['SalePrice'].index
cm = np.corrcoef(df_train[cols].values.T)
#print(cm)
sns.set(font_scale=1.25);
hm = sns.heatmap(cm, cbar=True, annot=True, square=True, fmt='.2f', annot_kws={'size': 10}, yticklabels=cols.values, xticklabels=cols.values)
#plt.show()
rem_cols = ['GarageArea', '1stFlrSf']

cols = [col for col in cols if col not in rem_cols]
sns.pairplot(df_train[cols], height=2.5)
#plt.show()

total = df_train.isnull().sum().sort_values(ascending=False)
percent = (df_train.isnull().sum() / df_train.isnull().count()).sort_values(ascending=False)

missing_data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent']);
#print(missing_data.head(20))

df_train = df_train.drop((missing_data[missing_data['Total'] > 1]).index, 1)
df_train['Electrical'] = df_train['Electrical'].fillna('Mix')
#print(df_train.info())
#df_train = df_train.drop(df_train.loc[df_train['Electrical'].isnull()].index)

#normalize(df_train, df_test)

salesPrice_scaled = StandardScaler().fit_transform(df_train['SalePrice'][:, np.newaxis]);
low_range = salesPrice_scaled[salesPrice_scaled[:, 0].argsort()][:10]
high_range = salesPrice_scaled[salesPrice_scaled[:, 0].argsort()][-10:]

sns.displot(df_train['SalePrice'], kde=True, color='b');
fig = plt.figure()
res = stats.probplot(df_train['SalePrice'], plot=plt)

df_train['SalePrice'] = np.log(df_train['SalePrice'])
sns.displot(df_train['SalePrice'], kde=True, color='b');
fig = plt.figure()
res = stats.probplot(df_train['SalePrice'], plot=plt)
#plt.show()

sns.displot(df_train['GrLivArea'], kde=True, color='b');
fig = plt.figure()
res = stats.probplot(df_train['GrLivArea'], plot=plt)

df_train['GrLivArea'] = np.log(df_train['GrLivArea'])
sns.displot(df_train['GrLivArea'], kde=True, color='b');
fig = plt.figure()
res = stats.probplot(df_train['GrLivArea'], plot=plt)

sns.displot(df_train['TotalBsmtSF'], kde=True, color='b');
fig = plt.figure()
res = stats.probplot(df_train['TotalBsmtSF'], plot=plt)

df_train['TotalBsmtSF'] = np.log(df_train['TotalBsmtSF'])
sns.displot(df_train['TotalBsmtSF'], kde=True, color='b');
fig = plt.figure()
res = stats.probplot(df_train['TotalBsmtSF'], plot=plt)

var = 'GrLivArea'
data = pd.concat([df_train['SalePrice'], df_train[var]], axis=1)
data.plot.scatter(x=var, y='SalePrice');

var = 'TotalBsmtSF'
data = pd.concat([df_train['SalePrice'], df_train[var]], axis=1)
data.plot.scatter(x=var, y='SalePrice');
plt.show()

print(salesPrice_scaled)
