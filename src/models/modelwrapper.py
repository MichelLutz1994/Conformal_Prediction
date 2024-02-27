class ModelWrapper:
    def __init__(self, model=None, type=None):
        self.model = model
        self.type = type
        self.acc = None

    def fit(self, x_train, y_train):
        self.classes_=set(y_train)
        self.sklearn_is_fitted=True
        self.model.fit(x_train, y_train)

    def predict(self, x_test):
        return self.model.predict(x_test)

    def predict_acc(self, x_test, y_test):
        y_pred = self.model.predict(x_test)
        acc = (y_pred == y_test).mean()
        self.acc = acc
        return acc

    def predict_proba(self, x_test):
        '''
        predict_proba wrapper for sklearn models
        :param x_test:
        :return: vector with class probabilities
        '''
        return self.model.predict_proba(x_test)