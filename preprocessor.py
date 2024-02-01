import pandas as pd
import numpy as np
import json
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=DeprecationWarning)


class Preprocessor:
    def __init__(self, path):
        self.data = pd.read_csv(path)
        self.path = path
        self.y = None
        self.features_to_drop = []

    def identify_mostly_empty_features(self):
        nans_freq = [i for i in self.data.isna().sum() / self.data.shape[0]]
        features_to_drop = list(self.data.columns[np.array(nans_freq) > 0.2])
        self.features_to_drop += features_to_drop

    def identify_highly_correlated(self):
        dd = self.data.corr().abs()
        s = dd.unstack()
        so = s.sort_values(kind="quicksort")
        corr = so[(s != 1) & (s >= 0.8)]

        to_drop = []
        for c in corr.index:
            if to_drop.count(c[0]) == 0:
                to_drop.append(c[1])

        self.features_to_drop += to_drop

    def fill_nans(self):
        # Fill all nan's with most common value of each feature
        for col in self.data.columns:
            nan_count = self.data[col].isna().sum()
            if nan_count != 0:
                self.data[col].fillna(self.data[col].value_counts().idxmax(), inplace=True)

    def fit(self):
        # If not test and "in-hospital_death" is missing terminate the execution
        if "In-hospital_death" not in self.data.columns:
            return "This data is not for Train. You need to run with test mode"

        # 1. Extract Labels
        self.y = self.data["In-hospital_death"]

        # 2. Extract Features having many Nan values
        self.identify_mostly_empty_features()
        # 3. Fill all nan's in data
        self.fill_nans()
        # 4. Identify highly correlated features
        self.identify_highly_correlated()

        tmp = set(self.features_to_drop)
        self.features_to_drop = list(tmp)

        self.save("features_to_drop.json")
        print(self.features_to_drop)

    def transform(self):
        if len(self.features_to_drop) == 0:
            self.load("features_to_drop.json")
        # 1. Delete Outcome-related Descriptors
        drop_lst = ['recordid', 'SAPS-I', 'SOFA', 'Length_of_stay', 'Survival', 'In-hospital_death']
        # Check presence of a column before dropping it
        for col in drop_lst:
            if col not in self.data.columns:
                drop_lst.remove(col)

        for col in drop_lst:
            if col in self.data.columns:
                self.data = self.data.drop(columns=col, axis=1)

        # 2. Drop unnecessary features \ identify features
        self.data = self.data.drop(columns=self.features_to_drop, axis=1)
        # 3. Fill all nan's in data
        self.fill_nans()

    def save(self, path):
        print(f"save mode:\n {len(self.features_to_drop)}")
        with open(path, 'w') as output_file:
            json.dump(self.features_to_drop, output_file)

    def load(self, path):
        f = open(path)
        self.features_to_drop = json.load(f)
        print(f"load mode:\n {len(self.features_to_drop)}")
        f.close()
