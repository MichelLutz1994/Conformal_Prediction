import numpy as np
from pyreadr import pyreadr
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler


class PermadData:
    def __init__(self):
        self.data = pyreadr.read_r('data/PERMAD/data.RData')['data']
        self.labs = pyreadr.read_r('data/PERMAD/labs.RData')['labs']
        self.time_data = pyreadr.read_r('data/PERMAD/time.RData')['time']
        self.sample_id = pyreadr.read_r('data/PERMAD/sample_id.RData')['sample_id']
        self.features_name = pyreadr.read_r('data/PERMAD/features.RData')['features']
        self.cohort = pyreadr.read_r('data/PERMAD/cohort.RData')['cohort']
        self.time_id = pyreadr.read_r('data/PERMAD/time_id.RData')['time_id']
        self.time_ct = pyreadr.read_r('data/PERMAD/time_ct.RData')['time_ct']

        # sort data by patient
        self.npdata = self.data.to_numpy()

        # normalize features
        #self.npdata = preprocessing.normalize(self.npdata, norm='l2', axis=0)
        scaler = StandardScaler()
        scaler.fit(self.npdata)
        self.npdata = scaler.transform(self.npdata)


        self.num_subjects = int(max(self.sample_id.to_numpy())[0])
        self.data_subjects = [self.npdata[:, (self.sample_id.to_numpy().T == subject)[0, :]] for subject in
                              np.unique(self.sample_id.to_numpy()).astype(int)]

        self.time_id_subjects = [self.time_id[(self.sample_id.to_numpy() == subject)] for subject in
                            np.unique(self.sample_id.to_numpy()).astype(int)]


    def __str__(self):
        return ("data: \t" + str(self.data.shape) + "\n" +
                "labs: \t" + str(self.labs.shape) + "\n" +
                "time: \t" + str(self.time_data.shape) + "\n" +
                "sample_id: \t" + str(self.sample_id.shape) + "\n" +
                "features: \t" + str(self.features_name.shape) + "\n" +
                "cohort: \t" + str(self.cohort.shape) + "\n" +
                "time_id: \t" + str(self.time_id.shape) + "\n" +
                "time_ct: \t" + str(self.time_ct.shape) + "\n")
