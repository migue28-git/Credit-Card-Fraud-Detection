def Grid_Search_func(method,X_train,y_train):
    from sklearn.model_selection import GridSearchCV

    if method == 'Logistic Regression':
        from sklearn.linear_model import LogisticRegression
        params = {'solver': ['newton-cg', 'lbfgs', 'sag', 'saga'], 'C': [0.1, 0.3, 0.7, 0.9, 1],
                  'max_iter': [100, 1000, 10000, 50000]}
        GS = GridSearchCV(LogisticRegression(), param_grid=params, n_jobs=6, scoring='roc_auc').fit(X_train,y_train)

    return GS