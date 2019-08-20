import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn import svm
from sklearn.metrics import roc_curve, roc_auc_score
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_selection import SelectFromModel
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.ensemble import (
    RandomForestClassifier,
    ExtraTreesClassifier,
    GradientBoostingClassifier,
    VotingClassifier,
    BaggingClassifier
)
from pandas.api.types import is_string_dtype, is_numeric_dtype


def get_model_results(model, x_train, y_train, x_test, y_test, model_name=None):
    model.fit(x_train, y_train)
    probs = model.predict_proba(x_test)[:, 1]
    if model_name:
        pass
    elif type(model).__name__ == 'GridSearchCV':
        model_name = type(model.estimator).__name__
    else:
        model_name = type(model).__name__
    print(f'\n{model_name}')
    auc = roc_auc_score(y_test, probs)
    print(f'AUC: {round(auc, 3)}')
    fpr, tpr, _ = roc_curve(y_test, probs)
    plot_roc_curve(fpr, tpr, model_name)


def plot_roc_curve(fpr, tpr, model_name):
    plt.plot(fpr, tpr, color='orange', label='ROC')
    plt.plot([0, 1], [0, 1], color='darkblue', linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'ROC Curve of {model_name}')
    plt.legend()
    plt.show()


DATA_PATH = 'WA_Fn-UseC_-Telco-Customer-Churn.csv'
ID_COL = 'customerID'
TARGET_COL = 'Churn'
TOTAL_CHARGES_COL = 'TotalCharges'
DUMMY_COLS = ['InternetService', 'PaymentMethod', 'Contract']
customer_data = pd.read_csv(DATA_PATH, index_col=ID_COL)

error_indices = pd.to_numeric(customer_data[TOTAL_CHARGES_COL], errors='coerce').isnull()
customer_data[TOTAL_CHARGES_COL] = pd.to_numeric(customer_data[TOTAL_CHARGES_COL], errors='coerce').fillna(0)

numerical_cols = [col for col in customer_data.columns if is_numeric_dtype(customer_data[col])]
string_cols = [col for col in customer_data.columns if is_string_dtype(customer_data[col]) and col not in DUMMY_COLS]

# Check some possible errors in our data

# Make sure we don't miss any columns
assert (set(numerical_cols + string_cols + DUMMY_COLS) == set(customer_data.columns))

# Make sure that we don't have any columns that might change into dummy variables with too many categories
assert (all([len(customer_data[col].unique()) < 10 for col in customer_data[string_cols]]))

for str_col in string_cols:
    print(str_col, customer_data[str_col].unique())

dummy_vars = pd.get_dummies(customer_data[DUMMY_COLS])
dummy_vars.columns = [col.replace(' ', '_') for col in dummy_vars.columns]

for col in string_cols:
    customer_data[col].replace(
        {'No': 0,
         'Yes': 1,
         'Female': 1,
         'Male': 0,
         'No phone service': 0,
         'No internet service': 0,
         },
        inplace=True,
    )

customer_data = customer_data[string_cols].join(dummy_vars)

X = customer_data.drop(TARGET_COL, axis=1)
y = np.array(customer_data[TARGET_COL].astype('int'))

# Some of the estimators require feature scaling to work properly
X_scaled = StandardScaler().fit_transform(X)

# Shuffle and divide the database into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=0)

# Setup crossvalidation
cv = 5

# Logistic Regression
tuned_parameters = {'C': [1, 10, 100, 1000]}
lrc = GridSearchCV(LogisticRegression(solver='lbfgs'), param_grid=tuned_parameters, n_jobs=-1, cv=cv)
get_model_results(lrc, X_train, y_train, X_test, y_test)

# Support Vector Machines
tuned_parameters = {'base_estimator__C': [1, 10, 100, 1000]}
svmc = GridSearchCV(BaggingClassifier(svm.SVC(gamma='auto'), max_samples=0.1, max_features=0.5),
                    param_grid=tuned_parameters, n_jobs=-1, cv=cv)
get_model_results(svmc, X_train, y_train, X_test, y_test, '(Bagged) SVM')

# Random Forest
tuned_parameters = {"max_depth": [3, 7, 11, 15]}
rfc = GridSearchCV(RandomForestClassifier(n_estimators=200), param_grid=tuned_parameters, cv=cv, n_jobs=-1)
get_model_results(rfc, X_train, y_train, X_test, y_test)

# Extra Trees Classifier
etc = GridSearchCV(ExtraTreesClassifier(n_estimators=200), param_grid=tuned_parameters, cv=cv, n_jobs=-1)
get_model_results(etc, X_train, y_train, X_test, y_test)

# XGBoost
gbm = GridSearchCV(GradientBoostingClassifier(n_estimators=200, learning_rate=0.05), param_grid=tuned_parameters, cv=cv,
                   n_jobs=-1)
get_model_results(gbm, X_train, y_train, X_test, y_test)

# Stacked classifier
lrc = Pipeline([
    ('models', FeatureUnion([
        ('rfc', SelectFromModel(rfc.best_estimator_)),
        ('etc', SelectFromModel(etc.best_estimator_)),
        ('gbm', SelectFromModel(gbm.best_estimator_)),
    ])),
    ('stacking', LogisticRegression(solver='lbfgs'))
])
get_model_results(lrc, X_train, y_train, X_test, y_test, 'Stacked Classifier')

# Voting classifier
vc = VotingClassifier(
    estimators=
    [
        ('lr', rfc.best_estimator_),
        ('rf', etc.best_estimator_),
        ('gbm', gbm.best_estimator_)
    ],
    voting='soft'
)
get_model_results(vc, X_train, y_train, X_test, y_test)

# Lets see which features were the most important for the best model
importances = rfc.best_estimator_.feature_importances_
std = np.std([tree.feature_importances_ for tree in rfc.best_estimator_.estimators_], axis=0)
indices = np.argsort(importances)

plt.title('Feature Importances')
plt.barh(range(len(indices)), importances[indices], color='b', align='center')
plt.yticks(range(len(indices)), [X.columns[i] for i in indices])
plt.xlabel('Relative Importance')
plt.show()
