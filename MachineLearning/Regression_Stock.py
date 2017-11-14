import pandas as pd
import quandl
import math, datetime
import numpy as np
import  matplotlib.pyplot as plt
from matplotlib import style
import pickle
from sklearn import preprocessing, svm
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

style.use('ggplot')

quandl.ApiConfig.api_key = 'vdYKgJkojqWAif1PwBYC'
df = quandl.get_table('WIKI/PRICES', ticker='GOOGL')
df = df[['date','adj_close', 'adj_volume']]
df = df.set_index('date')
#print(df.tail())

forecast_col = 'adj_close'
df.fillna(-99999, inplace=True)

forecast_out = int(math.ceil(0.01*len(df)))
df['label'] = df[forecast_col].shift(-forecast_out)
# print(df.head())

X = np.array(df.drop(['label'],axis=1))
X = preprocessing.scale(X)
X_predict = X[-forecast_out:]
X = X[:-forecast_out]

df.dropna(inplace=True)
y = np.array(df['label'])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4)

# clf = LinearRegression()
# clf.fit(X_train, y_train)
# with open('stock_lregression.pickle','wb') as fh:
#     pickle.dump(clf, fh)

pickle_in = open('stock_lregression.pickle', 'rb')
clf = pickle.load(pickle_in)

accuracy = clf.score(X_test, y_test)

forecast_set = clf.predict(X_predict)
print(forecast_set, accuracy, forecast_out)

df['Forecast'] = np.nan
df['adj_close'].plot()
df['Forecast'].plot()
last_date = df.iloc[-1].name
last_unix = last_date.timestamp()
one_day = 86400
next_unix = last_unix + one_day

for i in forecast_set:
    next_date = datetime.datetime.fromtimestamp(next_unix)
    next_unix += one_day
    df.loc[next_date] = [np.nan for _ in range(len(df.columns)-1)] + [i]

plt.clf()
df['adj_close'].plot()
df['Forecast'].plot()
plt.legend(loc=4)
plt.xlabel('Date')
plt.ylabel('Price')
plt.show()



