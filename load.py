import pandas as pd
from sklearn.preprocessing import StandardScaler


def load_data(choice):
    global data_name, X, y, k
    if choice == 1:
        url = r'E:\Code Files\Py code file\Graduation project\data\wine\wine.data'
        data = pd.read_csv(url, sep=',')
        X = data.iloc[:, 1:].values
        y = data.iloc[:, 0].values
        data_name = 'wine'
        k = 3
    elif choice == 2:
        url = r'E:\Code Files\Py code file\Graduation project\data\iris\iris.data'
        data = pd.read_csv(url, sep=',')
        X = data.iloc[:, :4].values
        y = data.iloc[:, 4].values
        data_name = 'iris'
        k = 3
    elif choice == 3:
        url = r'E:\Code Files\Py code file\Graduation project\data\seeds.CSV'
        data = pd.read_csv(url, sep=',')
        X = data.iloc[:, :7].values
        y = data.iloc[:, 7].values
        data_name = 'seeds'
        k = 3
    elif choice == 4:
        url = r'E:\Code Files\Py code file\Graduation project\data\Raisin_Dataset.csv'
        data = pd.read_csv(url, sep=',')
        X = data.iloc[:, :7].values
        y = data.iloc[:, 7].values
        data_name = 'Raisin'
        k = 2
    # 特征缩放
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    return X_scaled, y, data_name, k
