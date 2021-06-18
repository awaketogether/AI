import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns;
from sklearn.model_selection import learning_curve, ShuffleSplit, GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import roc_curve, auc

sns.set_theme()


class ROCModels(object):
    def __init__(self, model, title, y_pred=None, KNN_fpr=None, KNN_tpr=None, threshold=None, auc_knn=None):
        self.model = model
        self.y_pred = y_pred
        self.KNN_fpr = KNN_fpr
        self.KNN_tpr = KNN_tpr
        self.threshold = threshold
        self.auc_knn = auc_knn
        self.name = "KNN" + title

    def processRoc(self, X_test, X_train, y_train, y_test, pos_label='W'):
        self.model.fit(X_train, y_train)
        self.y_pred = self.model.predict_proba(X_test)

        self.KNN_fpr, self.KNN_tpr, self.threshold = roc_curve(y_test, self.y_pred[:, 1], pos_label=pos_label)
        self.auc_knn = auc(self.KNN_fpr, self.KNN_tpr)


def plotValidationModelCurves(estimator, title, X, y, axes=None, ylim=None, cv=None,
                              n_jobs=None, train_sizes=np.linspace(.1, 1.0, 5)):
    if axes is None:
        _, axes = plt.subplots(1, 3, figsize=(20, 5))

    axes[0].set_title(title)
    if ylim is not None:
        axes[0].set_ylim(*ylim)
    axes[0].set_xlabel("Training samples")
    axes[0].set_ylabel("Score")

    train_sizes, train_scores, test_scores, fit_times, _ = \
        learning_curve(estimator, X, y, cv=cv, n_jobs=n_jobs,
                       train_sizes=train_sizes,
                       return_times=True)
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    fit_times_mean = np.mean(fit_times, axis=1)
    fit_times_std = np.std(fit_times, axis=1)

    axes[0].grid()
    axes[0].fill_between(train_sizes, train_scores_mean - train_scores_std,
                         train_scores_mean + train_scores_std, alpha=0.1,
                         color="r")
    axes[0].fill_between(train_sizes, test_scores_mean - test_scores_std,
                         test_scores_mean + test_scores_std, alpha=0.1,
                         color="g")
    axes[0].plot(train_sizes, train_scores_mean, 'o-', color="r",
                 label="Training score")
    axes[0].plot(train_sizes, test_scores_mean, 'o-', color="g",
                 label="Cross-validation score")
    axes[0].legend(loc="best")

    axes[1].grid()
    axes[1].plot(train_sizes, fit_times_mean, 'o-')
    axes[1].fill_between(train_sizes, fit_times_mean - fit_times_std,
                         fit_times_mean + fit_times_std, alpha=0.1)
    axes[1].set_xlabel("Training examples")
    axes[1].set_ylabel("fit_times")
    axes[1].set_title("Scalability of the model")

    axes[2].grid()
    axes[2].plot(fit_times_mean, test_scores_mean, 'o-')
    axes[2].fill_between(fit_times_mean, test_scores_mean - test_scores_std,
                         test_scores_mean + test_scores_std, alpha=0.1)
    axes[2].set_xlabel("fit_times")
    axes[2].set_ylabel("Score")
    axes[2].set_title("Performance of the model")


def main():
    dataset = pd.read_csv("../../../SleepAnalyzer/outputAwakeInformation/sleepLogs.csv")
    dataset.drop(['Number', 'timestamp', 'count'], axis=1, inplace=True, errors='ignore')

    #INPUTS / OUTPUT
    x = dataset.iloc[:, :-1].values
    y = dataset.iloc[:, 3].values

    #SPLIT dataset in Input/output for the training and the testing.
    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.25)

    #Scaling the input for a better mathematical computation.
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Computation of the error for each training for a range of 1 to 40 for the hyper parameter 'k' of 'k-NN'.
    error = []
    worstKNNModels = []
    KNNErrorsPlot = 0

    for i in range(1, 25):
        knn = KNeighborsClassifier(n_neighbors=i)
        knn.fit(X_train, y_train)
        pred_i = knn.predict(X_test)
        if np.mean(pred_i != y_test) >= 0.068:
            worstKNNModels.append(knn)
        error.append(np.mean(pred_i != y_test))

    # Plot k-NN errors for the range 1 to 40
    if KNNErrorsPlot == 1:
        plt.figure(figsize=(12, 6))
        plt.plot(range(1, 25), error, color='red', linestyle='dashed', marker='o',
                 markerfacecolor='blue', markersize=10)
        plt.title('Error Rate K Value')
        plt.xlabel('K Value')
        plt.ylabel('Mean Error')
        plt.show()

    # Comparison of the best and the worst classifer with:
    # - Their learning curves
    # - Their AUROC curves

    print(error.index(min(error)))
    print(min(error))
    classifierOptimised = KNeighborsClassifier(n_neighbors=error.index(min(error)))
    classifierWithoutOptimisation = KNeighborsClassifier(n_neighbors=error.index(max(error)))
    learningCurve = 0
    ROC = 1

    # Learning curves is a good metric to understand whether the model is overfitting/goodFitting/underFitting
    if learningCurve == 1:
        fig, axes = plt.subplots(3, 2, figsize=(20, 15))
        fig.suptitle('K-NN Bad/Good fit check')

        cv = ShuffleSplit(n_splits=100, test_size=0.25, random_state=0)
        plotValidationModelCurves(estimator=classifierOptimised, title="Learning Curves Good Fit Model", X=x, y=y,
                                  axes=axes[:, 0], ylim=(0.7, 1.01), cv=cv, n_jobs=4)
        plotValidationModelCurves(estimator=classifierWithoutOptimisation, title="Learning Curves bad Fit Model", X=x, y=y,
                                  axes=axes[:, 1], ylim=(0.7, 1.01), cv=cv, n_jobs=4)
        plt.show()

    # ROC is a very good metric to judge the efficiency and sensitivy of a model in terms of true positive and false postive rate
    # The following shows the best model with the worst ones in order to also have an idea of the gap made in optimising
    # only one hyper parameter: 'k'.
    if ROC == 1:
        models = []
        idx = 0

        #WORST MODELS
        for x in worstKNNModels:
            model = ROCModels(x, str(idx))
            model.processRoc(X_test, X_train, y_train, y_test)
            models.append(model)
            idx += 1

        #BEST MODEL
        classifierOptimised.fit(X_train, y_train)
        classifierOptimised_y_pred = classifierOptimised.predict_proba(X_test)
        optimised_KNN_fpr, optimised_KNN_tpr, thresholdOpt = roc_curve(y_test, classifierOptimised_y_pred[:, 1], pos_label='W')
        auc_optimised_knn = auc(optimised_KNN_fpr, optimised_KNN_tpr)

        #PLOT of all ROC CURVES
        plt.figure(figsize=(15, 8), dpi=100)

        for x in models:
            plt.plot(x.KNN_fpr, x.KNN_tpr, linestyle='-', label='KNN Model' + x.name + '(auc = %0.3f' % x.auc_knn)

        plt.plot(optimised_KNN_fpr, optimised_KNN_tpr, linestyle='-', label='KNN Model OPTIMISED(auc = %0.3f' % auc_optimised_knn)
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.legend()
        plt.show()


if __name__ == "__main__":
    main()
