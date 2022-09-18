def plot_data_binary(X, y,l):
    import matplotlib.pyplot as plt
    plt.scatter(X[y == 0, 0], X[y == 0, 1], label="Class #0", alpha=0.5, linewidth=0.15)
    plt.scatter(X[y == 1, 0], X[y == 1, 1], label="Class #1", alpha=0.5, linewidth=0.15, c='y')
    plt.legend()
    plt.title(l)
    return plt.show()