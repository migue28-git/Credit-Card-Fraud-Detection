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
