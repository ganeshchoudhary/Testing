import pandas as pd
import sklearn
from sklearn import linear_model
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn import metrics

def read_data_set():
    df = pd.read_csv("training_dataset.csv")
    df.dropna()
    print(df.shape)
    print(df['y'].isnull().values.any())
    df.dropna(subset=["y"], inplace=True)
    print(df['y'].isnull().values.any())
    print(df.describe())
    features = df[['x']]
    target = df['y']
    features_train, features_test, target_train, target_test= train_test_split(features, target, test_size=0.4)
    reg = linear_model.LinearRegression()
    reg.fit(features_train, target_train)
    predictions=reg.predict(features_test)
    plt.scatter(target_test, predictions)
    # plt.show()
    print(metrics.mean_absolute_error(target_test, predictions))
    print(metrics.mean_squared_error(target_test, predictions))


if __name__ == '__main__':
    read_data_set()
