import numpy as np
import pandas as pd

def plot_data_binary(X, y,l):
    import matplotlib.pyplot as plt
    plt.scatter(X[y == 0, 0], X[y == 0, 1], label="Class #0", alpha=0.5, linewidth=0.15)
    plt.scatter(X[y == 1, 0], X[y == 1, 1], label="Class #1", alpha=0.5, linewidth=0.15, c='y')
    plt.legend()
    plt.title(l)
    return plt.show()
def plot_PCA_binary(df,t='Plot'):
    from sklearn.decomposition import PCA
    import mglearn
    import matplotlib.pyplot as plt
    # Nomalize data
    from sklearn.preprocessing import MinMaxScaler
    df_normalized = MinMaxScaler().fit_transform(df.copy())
    # Create a PCA instance with 2 components:
    pca_features = PCA(n_components=2).fit_transform(df_normalized[:, :-1])
    mglearn.discrete_scatter(pca_features[:, 0], pca_features[:, 1], df_normalized[:, -1])
    plt.legend(['0', '1'])
    plt.xlabel('PCA1')
    plt.ylabel('PCA2')
    plt.title(t)
    return plt.show()

#Data
df = pd.read_csv('creditcard.csv')

#Describing data and visualizating proportion and PCA tansformation
df.info()
plot_data_binary(df,'0-1 Proportion')
plot_PCA_binary(df.copy().drop('Time',axis=1),'PCA data transformation')

#Since date is highly unbalanced, we use a SMOTE method for re-balance


