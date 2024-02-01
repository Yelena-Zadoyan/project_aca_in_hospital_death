import json
from model import Model
from preprocessor import Preprocessor
from sklearn.model_selection import train_test_split

class Pipeline:
    def __init__(self, path, test=False):
        self.preprocessor = Preprocessor(path=path)
        self.model = Model()
        self.test = test

    def run(self):
        if self.test:
            # call preprocessor and model for testing
            self.preprocessor.transform()
            _, y_test_proba = self.model.predict(self.preprocessor.data)
            # print({"Predict_probas": list(y_test_proba[:, 1]), "threshold": 0.5})
            probas_dct = {"Predict_probas": list(y_test_proba[:, 1]), "threshold": 0.5}
            with open("predictions.json", "w") as outfile:
                json.dump(probas_dct, outfile)
        else:
            # call preprocessor and model for training
            self.preprocessor.fit()
            # Check if label was processed properly, if not terminate with a message
            self.preprocessor.transform()
            y = self.preprocessor.y
            X = self.preprocessor.data
            self.model.fit(X, y)
            y_pred, _ = self.model.predict(X)

            print(self.model.get_scores(y_real=y, y_pred=y_pred))



