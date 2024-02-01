import pickle
from sklearn.ensemble import BaggingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score

class Model:
    def __init__(self):
        self.model = None
        self.filename = "model.sav"

    def fit(self, X, y):
        self.model = BaggingClassifier(
            LogisticRegression(penalty='l1', solver='liblinear', max_iter=10000, class_weight="balanced"),
            n_estimators=10, random_state=0)
        self.model.fit(X, y)
        pickle.dump(self.model, open(self.filename, 'wb'))

    def predict(self, X):
        if self.model is None:
            self.model = pickle.load(open(self.filename, 'rb'))

        return self.model.predict(X), self.model.predict_proba(X)

    def get_scores(self, y_real, y_pred):
        accuracy = accuracy_score(y_real, y_pred)
        f1 = f1_score(y_real, y_pred)
        precision = precision_score(y_real, y_pred)
        recall = recall_score(y_real, y_pred)
        auc_score = roc_auc_score(y_real, y_pred)

        results = {"accuracy": accuracy, "f1": f1, "precision": precision, "recall": recall, "auc_score": auc_score}

        return results
