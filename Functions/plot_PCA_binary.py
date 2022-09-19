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
