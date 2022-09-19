import numpy as np
import pandas as pd

def plot_data_binary(df,t='Plot'):
    import matplotlib.pyplot as plt
    X = df.copy().iloc[:, :-1].values
    y = df.copy().iloc[:, -1].values
    balanced_ratio = df.iloc[:, -1].value_counts()
    balanced_ratio = str(round((balanced_ratio[1] / balanced_ratio[0] * 100),4)) + '%'  # % of frauds
    plt.figure()
    plt.scatter(X[y == 0, 0], X[y == 0, 1], label="Class #0", alpha=0.5, linewidth=0.15)
    plt.scatter(X[y == 1, 0], X[y == 1, 1], label="Class #1", alpha=0.5, linewidth=0.15, c='y')
    plt.legend()
    plt.suptitle(balanced_ratio+' of positive instances',style='italic')
    plt.title(t)
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
    plt.figure()
    mglearn.discrete_scatter(pca_features[:, 0], pca_features[:, 1], df_normalized[:, -1])
    plt.legend(['0', '1'])
    plt.xlabel('PCA1')
    plt.ylabel('PCA2')
    plt.title(t)
    return plt.show()

#Data
df = pd.read_csv('creditcard.csv')
X = df.copy().iloc[:,:-1].drop('Time',axis=1)
y = df.copy().iloc[:,-1]

#Describing data and visualizating proportion and PCA tansformation
df.info()
plot_data_binary(df.copy(),'0-1 Proportion')
plot_PCA_binary(df.copy().drop('Time',axis=1),'PCA data transformation')

#Since data is highly unbalanced, we use a SMOTE method for re-balance
from imblearn.over_sampling import SMOTE
X_resampled,y_resampled = SMOTE(sampling_strategy=40/100,random_state=0).fit_resample(X,y)
df_resampled = pd.concat([X_resampled,y_resampled],axis=1,join='inner')
plot_data_binary(df_resampled,'0-1 Proportion SMOTE Method')
plot_PCA_binary(df_resampled,'PCA data transformation SMOTE Method')




