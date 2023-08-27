import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import numpy as np
from collections import defaultdict
import tensorflow as tf;
from sklearn.preprocessing import StandardScaler
from scipy import stats

train_data = pd.read_csv('./home_data/train.csv')
test_data = pd.read_csv('./home_data/test.csv')

df_train = train_data.copy()
df_test = test_data.copy()
salesPrice = df_train['SalePrice']
def plotData(data, var):
    data = pd.concat([data['SalePrice'], data[var]], axis=1)
    data.plot.scatter(x=var, y='SalePrice', ylim=(0, 800000));
    plt.show()

def displot(data, var):
    sns.displot(data[var], kde=True, color='b');
    plt.figure()
    res = stats.probplot(data[var], plot=plt)
    plt.show()

# plotData(df_train, 'GrLivArea')
# plotData(df_train, 'TotalBsmtSF')
# plotData(df_train, 'OverallQual')

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
#df_train = df_train.drop(df_train.loc[df_train['Electrical'].isnull()].index)

salesPrice_scaled = StandardScaler().fit_transform(df_train['SalePrice'][:, np.newaxis]);
low_range = salesPrice_scaled[salesPrice_scaled[:, 0].argsort()][:10]
high_range = salesPrice_scaled[salesPrice_scaled[:, 0].argsort()][-10:]

df_train['SalePrice'] = np.log(df_train['SalePrice'])
df_train['GrLivArea'] = np.log(df_train['GrLivArea'])
df_train['TotalBsmtSF'] = np.log(df_train['TotalBsmtSF'])

displot(df_train, 'SalePrice')
displot(df_train, 'GrLivArea')
displot(df_train, 'TotalBsmtSF')
plotData(df_train, 'GrLivArea')
plotData(df_train, 'TotalBsmtSF')

print(salesPrice_scaled)
