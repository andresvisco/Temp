try:
    from sklearn.impute import SimpleImputer as Imputer
except ImportError:
    from sklearn.preprocessing import Imputer
import joblib

imputer = joblib.load('Train_imputer_noshow.sav')
exported_pipeline = joblib.load("Train_model.sav")
print(exported_pipeline)