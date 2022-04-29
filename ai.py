import math

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, ConfusionMatrixDisplay, log_loss
from sklearn.metrics import plot_confusion_matrix
from tqdm import tqdm

import wandb

def nn(
        data: pd.DataFrame,
        test: pd.DataFrame,
        testY: pd.DataFrame
    ):

    # inits the W&B serice
    #wandb.init(project="Titanic data set")

    # defines model
    mlp_clf = MLPClassifier(hidden_layer_sizes=(10, 10),
                            max_iter=1,
                            activation='relu',
                            solver='adam')

    testAcc = None
    y_pred = None
    testAccHist = []
    knownAcc = None
    cEnt = None
    n_classes = np.unique(data['Survived'])

    for x in range(0, 1000):
        mlp_clf.partial_fit(data.drop('Survived', axis=1), data['Survived'], classes=n_classes)

        # test model
        y_pred = mlp_clf.predict(test)
        y_data = mlp_clf.predict(data.drop('Survived', axis=1))

        # cross-entropy loss
        cEnt = log_loss(testY, y_pred)

        # return metrics
        testAcc = accuracy_score(testY, y_pred)
        knownAcc = accuracy_score(data['Survived'], y_data)

        testAccHist.append(testAcc)

        #wandb.log({"test Acc": testAcc})
        #wandb.log({"cEnt test loss": cEnt})
        # wandb.log({"known Acc": knownAcc})

        # early stop training if test acc is going down hill
        if len(testAccHist) > 10:
            if testAccHist[-10] > testAccHist[-1]:
                break


    #print("size", len(testAccHist))
    print('Accuracy test: {:.2f}'.format(testAcc))
    print('cross-entropy loss: {:.2f}'.format(cEnt))
    print('Accuracy known: {:.2f}'.format(knownAcc))

    # plot graph to show performance of model
    fig = ConfusionMatrixDisplay.from_predictions(testY, y_pred, display_labels=['dead', 'alive'])
    fig.figure_.suptitle("Confusion Matrix for Titanic Dataset")
    plt.show()

# graphs out the av last test error for 30 runs with varying early stopping params
def earlyStoppingTuning(
        data: pd.DataFrame,
        test: pd.DataFrame,
        testY: pd.DataFrame
    ):

    # inits the W&B serice
    wandb.init(project="Titanic data set")

    # defines model
    mlp_clf = MLPClassifier(hidden_layer_sizes=(10, 10),
                            max_iter=1,
                            activation='relu',
                            solver='adam')

    testAcc = None
    testAccHist = []
    testAccHistLen = [x for x in range(1, 30)]
    knownAcc = None
    cEnt = None
    n_classes = np.unique(data['Survived'])

    for histLen in testAccHistLen:
        histLenPerf = []
        # 30 runs to get the average performance for the early stopping
        for y in tqdm(range(0, 2)):
            # train
            for x in range(0, 1000):
                mlp_clf.partial_fit(data.drop('Survived', axis=1), data['Survived'], classes=n_classes)

                # test model
                y_pred = mlp_clf.predict(test)
                y_data = mlp_clf.predict(data.drop('Survived', axis=1))

                # cross-entropy loss
                cEnt = log_loss(testY, y_pred)

                # return metrics
                testAcc = accuracy_score(testY, y_pred)
                knownAcc = accuracy_score(data['Survived'], y_data)

                testAccHist.append(testAcc)

                # wandb.log({"test Acc": testAcc})
                # wandb.log({"cEnt loss": cEnt})
                # wandb.log({"known Acc": knownAcc})

                # early stop training if test acc is going down hill
                if len(testAccHist) > histLen:
                    if testAccHist[-histLen] > testAccHist[-1]:
                        break
            # get testAcc and append it to histLenPerf
            histLenPerf.append(testAcc)
        # once its run 30 times average it and return the results
        wandb.log({"Early stopping performance": sum(histLenPerf)/len(histLenPerf)})
