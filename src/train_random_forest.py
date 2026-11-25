import pandas as pd
import numpy as np
import joblib
from preprocessing import Preprocessor
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

X_train, X_test, y_train, y_test = Preprocessor.load_and_split_data()
rfc = RandomForestClassifier(n_estimators= 100, random_state= 42)
rfc.fit(X_train,y_train)
y_pred = rfc.predict(X_test)
print(accuracy_score(y_pred, y_test))
print(confusion_matrix(y_test, y_pred))

joblib.dump(rfc, "models/test_gesture_model.joblib")