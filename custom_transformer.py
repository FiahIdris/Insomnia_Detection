from sklearn.base import TransformerMixin
import numpy as np


def bmi_function(value):
    description = 'healthy'
    if value >= 18.5 and value <= 24.9:
        description = 'healthy'
    elif value >= 25 and value <= 29.9:
        description = 'overweight'
    elif value <= 18.5:
        description = 'underweight'
    elif value >= 30:
        description = 'obese'
    return description

# Function for bp description based on data from wikipedia.


def blood_pressure_description(ubp, lbp):
    description = 'normal'
    if ubp <= 200 and lbp <= 80:
        description = 'normal'
    elif ubp >= 120 and ubp <= 129 and lbp <= 80:
        description = 'elevated'
    elif (ubp >= 30 and ubp <= 139) or (lbp >= 80 and lbp <= 89):
        description = 'hypertention_1'
    elif (ubp >= 140 and ubp <= 180) or (lbp >= 90 and lbp <= 120):
        description = 'hypertention_2'
    elif ubp > 180 or lbp >= 120:
        description = 'crisis'
    return description

# Create custom transformer for new column.


class NewColumnTransform(TransformerMixin):
    def fit(self, X, y=None, **fit_params):
        return self

    def transform(self, X, y=None, **fit_params):
        # Create BMI column
        bmi = X['weight']/np.square(X['height']/100)
        X['bmi'] = bmi.tolist()
        # Create bmi_description column
        X['bmi_description'] = X.bmi.apply(lambda x: bmi_function(x))
        # Create bp_description column
        X['bp_description'] = X.apply(
            lambda x: blood_pressure_description(x.ubp, x.lbp), axis=1)
        # Combaining column pernicious_1 and pernicious_2.
        X['bad_habit'] = X['pernicious_1'] + X['pernicious_2']
        X.drop(['pernicious_1', 'pernicious_2'], axis=1, inplace=True)
        return X
