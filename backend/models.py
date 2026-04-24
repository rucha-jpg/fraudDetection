class WrappedModel:
    def __init__(self, scaler, clf):
        self.scaler = scaler
        self.clf    = clf
    def predict_proba(self, X):
        return self.clf.predict_proba(self.scaler.transform(X))

class WrappedExplainer:
    def __init__(self, explainer, scaler):
        self.explainer = explainer
        self.scaler    = scaler
    def shap_values(self, X):
        return self.explainer.shap_values(self.scaler.transform(X))