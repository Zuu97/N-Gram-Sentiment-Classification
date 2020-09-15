import os
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from variables import*
from util import get_sentiment_data

np.random.seed(seed)

from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score

class nGramClassifier(object):
    def __init__(self):
        Xtrain, Ytrain, Xval, Yval, Xtest, XtestOrg = get_sentiment_data()
        self.Xtrain = Xtrain
        self.Ytrain = Ytrain
        self.Xval = Xval
        self.Yval = Yval
        self.Xtest = Xtest
        self.XtestOrg = XtestOrg

    def RandomForest(self):
        self.model = RandomForestClassifier(max_depth=depth, random_state=seed)
        self.model.fit(self.Xtrain, self.Ytrain)

    def SupportVectorMachine(self):
        self.model = SVC(C=C, kernel='rbf', random_state=seed)
        self.model.fit(self.Xtrain, self.Ytrain)

    def kNearestNeighbour(self):
        self.model = KNeighborsClassifier(n_neighbors=n_neighbors)
        self.model.fit(self.Xtrain, self.Ytrain)

    def model_evaluation(self):
        YpredTrain = self.model.predict(self.Xtrain)
        YpredVal = self.model.predict(self.Xval)

        Train_acc = round(accuracy_score(self.Ytrain, YpredTrain), 3)
        Val_acc   = round(accuracy_score(self.Yval, YpredVal), 3)

        Train_pre = round(precision_score(self.Ytrain, YpredTrain), 3)
        Val_pre   = round(precision_score(self.Yval, YpredVal), 3)

        Train_rec = round(recall_score(self.Ytrain, YpredTrain), 3)
        Val_rec   = round(recall_score(self.Yval, YpredVal), 3)

        print("             ---------------------------------------------")
        print("             |   Accuracy   |   Precision   |   Recall   |")
        print("----------------------------------------------------------")
        print("  Training   |     %.3f    |      %.3f    |    %.3f   |"% (Train_acc , Train_pre , Train_rec))
        print("----------------------------------------------------------")
        print(" Validation  |     %.3f    |      %.3f    |    %.3f   |"% (Val_acc , Val_pre , Val_rec))
        print("----------------------------------------------------------")

    def save_model(self):
        with open(pkl_filename, 'wb') as file:
            pickle.dump(self.model, file)

    def load_model(self):
        with open(pkl_filename, 'rb') as file:
            self.model = pickle.load(file)

    def model_predictions(self):
        Ypred = self.model.predict(self.Xtest)
        Ypred = [pred_dict[sign] for sign in Ypred]
        XtestOrg = self.XtestOrg.tolist()
        data = {'Recommendation':Ypred, 'Review':XtestOrg}
        df = pd.DataFrame(data)
        df.to_csv(csv_path, index=False)

    def run(self):
        if not os.path.exists(pkl_filename):
            print("{} Classifier Saving".format(model_name))
            self.SupportVectorMachine()
            self.save_model()
        else:
            print("{} Classifier Loading".format(model_name))
            self.load_model()
        self.model_evaluation()
        self.model_predictions()

if __name__ == "__main__":
    model = nGramClassifier()
    model.run()