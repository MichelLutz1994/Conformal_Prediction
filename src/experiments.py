from sklearn.model_selection import train_test_split

from data.data_generation import create_artificial_data, load_beans_data
from models.myMLP import MyMLP

from conformal_methods.byhand_evaluation import score_methode

from data.data_generation import create_artificial_data
from models.traditional_models import GaussianNBModel, SVM, RandomForest, AdaBoost


dataset="beans"
#dataset="artifical"
if dataset=="beans":
    data, classes = load_beans_data()
    n_classes = len(classes.classes_)
    n_features = data.shape[1]
else:
    n_classes = 5
    n_features = 20
    data = create_artificial_data(n_features, n_classes)

print('Shape of the data:', data.shape)
print("Classes: ", n_classes, " Features : ", n_features)

Y = data["Class"]
X = data.drop("Class", axis=1)
X_train, X_rest1, y_train, y_rest1 = train_test_split(X, Y, train_size=7000, random_state=2)
X_test, X_rest2, y_test, y_rest2 = train_test_split(X_rest1, y_rest1, train_size=1000, random_state=42)
X_calib, X_new, y_calib, y_new = train_test_split(X_rest2, y_rest2, train_size=1000, random_state=42)

mlp_model = MyMLP(5, 20)
mlp_model.fit(X_train, y_train)
result_mlp = mlp_model.predict_proba(X_test)


