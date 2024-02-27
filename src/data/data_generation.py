import zipfile
from os.path import exists

import numpy as np
import pandas as pd

from sklearn.datasets import make_classification
from sklearn.preprocessing import LabelEncoder


def create_artificial_data(n_features, n_classes, sample_size=10000, random_stat=42):
    X, Y = make_classification(
        n_samples=sample_size,
        n_features=n_features,
        n_informative=15,
        n_redundant=2,
        n_repeated=0,
        n_classes=n_classes,
        n_clusters_per_class=1,
        weights=None,
        flip_y=0.001,
        class_sep=1.0,
        hypercube=True,
        shift=0.0,
        scale=2.0,
        shuffle=True,
        random_state=random_stat)

    data = pd.DataFrame(X, columns=['X_ %i' % i for i in range(20)])
    data['Class'] = Y
    return data

def load_beans_data():
    # import dataset
    bean_data_file = "./data/DryBeanDataset/dry_bean_dataset.zip"
    if exists(bean_data_file):
        with zipfile.ZipFile(bean_data_file, 'r') as zip_ref:
            zip_ref.extractall(path="../data")
    bean_data_file = "./data/DryBeanDataset/Dry_Bean_Dataset.xlsx"
    beans = pd.read_excel(bean_data_file, engine='openpyxl')

    le = LabelEncoder()
    beans["Class"] = le.fit_transform(beans["Class"])
    return beans, le