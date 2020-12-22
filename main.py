import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import BaggingRegressor
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error

df = pd.read_csv("house_prices.csv")

bins = list(range(0, df["Price"].max(), 100000))
bin_labels = ["{} - {}".format(bins[i - 1], j) for i,j in enumerate(bins[1:])]
df["Binned"] = pd.cut(df["Price"], bins, labels=bin_labels)

# EDA
fig, ax = plt.subplots()
plt.title("Distribution of homes under 500k")
ax.violinplot(df[df["Price"] < 500000]["Price"])
plt.ylabel("Price")
plt.xlabel("Kernel Density")
plt.show()

fig, ax = plt.subplots()
plt.title("Distribution of homes over 500k")
ax.violinplot(df[df["Price"] > 500000]["Price"])
plt.ylabel("Price")
plt.xlabel("Kernel Density")
plt.show()

df_less_than_500 = df[df["Price"] > 500000]
df_less_than_500_counts = df_less_than_500['Binned'].value_counts()[df_less_than_500['Binned'].value_counts() > 0]
fig,ax = plt.subplots(figsize=(15, 10))
plt.title("Number of listings for different price ranges")
ax.bar(df_less_than_500_counts.index, df_less_than_500_counts.tolist())
plt.setp(ax.get_xticklabels(), rotation=90)
plt.show()

# Preprocessing
df = df[df["Type"] != "Apartment"]
df["Type"] = CountVectorizer().fit_transform(df["Type"].tolist()).toarray().tolist()
df["Type"] = df["Type"].apply(lambda x: x.index(1))
    
corr = df.corr()

# model building
X, y = df.iloc[:, 1:-1].to_numpy(), df["Price"].to_numpy()
x_train, x_test, y_train, y_test = train_test_split(X, y)

lr_model = LinearRegression().fit(x_train,y_train)

params = {"n_estimators" : range(10, 100, 10)}
bagging = GridSearchCV(BaggingRegressor(), params).fit(x_train,y_train).best_estimator_

params = {'criterion': ('mse', 'mae'), 
          'max_features': ('auto', 'sqrt', 'log2')}
randomforest = GridSearchCV(RandomForestRegressor(), 
                      params).fit(x_train, y_train).best_estimator_

print(lr_model.score(x_test, y_test))
print(bagging.score(x_test, y_test))
print(randomforest.score(x_test, y_test))