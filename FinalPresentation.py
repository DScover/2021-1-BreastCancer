# -*- coding: utf-8 -*-
# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.11.4
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# + id="GkAM_UGk_55S"
import xgboost as xgb
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# + id="s0rQ45cw_55V"
import sklearn
from sklearn.preprocessing import StandardScaler, OrdinalEncoder, OneHotEncoder
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.metrics import plot_confusion_matrix, accuracy_score ,roc_curve, auc, plot_roc_curve
from sklearn.compose import ColumnTransformer
from xgboost import XGBClassifier, XGBRegressor


# + [markdown] id="05UsbnIo_55Y"
# # 암 종류 Classify

# + [markdown] id="O3bEgXEB_55Y"
# ### Data Load

# + id="OTXcuV1I_55Y" outputId="c4485cb1-fc8e-4706-8a7e-fc7641ed9a62"
df=pd.read_csv('METABRIC_RNA_Mutation.csv')  # raw_data
mut=pd.read_csv('mutation.csv')   # 기연 언니가 처리한 mutation data; 0이 아닌 mutant type -> 1/ 나머지 -> 0
data=pd.concat([df.iloc[:,:31],mut], axis=1)   
data

# + id="fsSHq_0J_55Z" outputId="b32d1de0-9435-41de-b4ca-33122a465d63"
pd.set_option('display.max_rows', 700)
pd.set_option('display.max_columns', 700)
df.isnull().sum(axis=0)

# + [markdown] id="ffvO1Ihw_55a"
# ### Preprocessing

# + id="YZtTXep0_55a"
# 결측값 NaN 제거
drop_col=['patient_id', 'cancer_type','tumor_stage','3-gene_classifier_subtype', 'primary_tumor_laterality']
drop_data=data.drop(drop_col,axis=1)
drop_data=drop_data.dropna()

# X, y 분리
X=drop_data.copy()
y=X.pop('cancer_type_detailed')

# train/test set split
X_train, X_test, y_train, y_test=train_test_split(X,y,
                                                 test_size=0.2, 
                                                 random_state=1234, 
                                                 shuffle=True,
                                                 stratify=y)

# Encoding & Scale
ordinal_cols=['cellularity']
standard_cols=['age_at_diagnosis', 'lymph_nodes_examined_positive','mutation_count','nottingham_prognostic_index', 'overall_survival_months','tumor_size']
onehot_cols=['type_of_breast_surgery','pam50_+_claudin-low_subtype','cohort', 'er_status_measured_by_ihc','er_status', 'her2_status_measured_by_snp6','her2_status','tumor_other_histologic_subtype','inferred_menopausal_state','integrative_cluster', 'oncotree_code', 'pr_status', 'death_from_cancer']

onehot=Pipeline([
    ('oridinal', OrdinalEncoder()),
    ('onehot', OneHotEncoder())
])
column_trans=ColumnTransformer([
    ('stadardard', StandardScaler(), standard_cols),
    ('ordinal', OrdinalEncoder(categories=[['Low','Moderate','High']]), ordinal_cols),
    ('onehot', onehot, onehot_cols)],
    remainder="passthrough"
)

scaled_X_train=column_trans.fit_transform(X_train)
scaled_X_test=column_trans.transform(X_test)

# + [markdown] id="rVT4X74I_55a"
# ### Learning & Predict

# + id="QwlLHXKw_55b" outputId="313e26f0-7672-4122-81fc-87e6a207b1f7"
clf=XGBClassifier()
clf.fit(scaled_X_train, y_train)
pred=clf.predict(scaled_X_test)

# + id="D_o8rTxg_55b" outputId="b14ec368-98c3-40a3-a3ec-62a072cbd0af"
plot_confusion_matrix(estimator=clf, X=scaled_X_test,y_true=y_test)

# + [markdown] id="aT-wtPzS_55b"
# # Overall Survival

# + id="JzUcA6TN_55d"
# column 생성
onehot=[]
for col in  onehot_cols:
    onehot=onehot+sorted(list(set(X_train[col])))
mutation=list(mut.columns)
columns=standard_cols + ordinal_cols + onehot + ['chemotherapy', 'neoplasm_histologic_grade','hormone_therapy','radio_therapy'] + mutation

# + id="2rAsNtKm_55d"
# Drop Nan
drop_col=['patient_id', 'cancer_type','tumor_stage','3-gene_classifier_subtype', 'primary_tumor_laterality', 'overall_survival_months', 'death_from_cancer']
drop_data=data.drop(drop_col,axis=1)
drop_data=drop_data.dropna()

# X, y split
y=drop_data.pop('overall_survival')
X=drop_data.copy()

# train/test split
X_train, X_test, y_train, y_test=train_test_split(X,y,
                                                 test_size=0.2, 
                                                 random_state=1234, 
                                                 shuffle=True,
                                                 stratify=y)
# Encoding & Scale
ordinal_cols=['cellularity']
standard_cols=['age_at_diagnosis', 'lymph_nodes_examined_positive','mutation_count','nottingham_prognostic_index','tumor_size']
onehot_cols=['type_of_breast_surgery','cancer_type_detailed','pam50_+_claudin-low_subtype','cohort', 'er_status_measured_by_ihc','er_status', 'her2_status_measured_by_snp6','her2_status','tumor_other_histologic_subtype','inferred_menopausal_state','integrative_cluster', 'oncotree_code', 'pr_status']

onehot=Pipeline([
    ('oridinal', OrdinalEncoder()),
    ('onehot', OneHotEncoder())
])
column_trans=ColumnTransformer([
    ('stadardard', StandardScaler(), standard_cols),
    ('ordinal', OrdinalEncoder(categories=[['Low','Moderate','High']]), ordinal_cols),
    ('onehot', onehot, onehot_cols)],
    remainder="passthrough"
)

pipe=Pipeline([
    ('preprocess', column_trans),
    ('clf', XGBClassifier(random_state=1234,
                          eval_metric='logloss',
                          objective='binary:logistic',
                          n_estimators=1000,
                          max_depth=8,
                          min_child_weight=3,
                          learning_rate=0.0001,
                          colsample_bylevel=0.9,
                          colsample_bytree=0.4,
                          gamma=0
                         ))])

# + id="hgzXEQWw_55d" outputId="3c5482af-f58d-455e-be47-269e52ba398d"
pipe.fit(X_train, y_train)

# + id="sFrHOyCZ_55e"
pred=pipe.predict(X_test)

# + id="d1MyL3Id_55e"
#pd.DataFrame(scaled_X_train, columns=columns).to_csv('data/X_train_with_col.csv')
#pd.DataFrame(scaled_X_test, columns=columns).to_csv('data/X_test_with_col.csv')

# + id="1gHVB8gp_55f" outputId="48208efb-16df-4532-f670-7e4c686db8b1"
plot_confusion_matrix(estimator=pipe, X=X_test,y_true=y_test)

# + id="Fs6fPjE8_55g" outputId="cf92dbe1-4d14-4422-c7b1-e00b19687e94"
accuracy_score(y_true=y_test, y_pred=pred)

# + id="Vy-yexdw_55g"
from sklearn.metrics import precision_score, recall_score

# + id="8_LeXmsG_55g" outputId="d94af846-ede8-487b-a410-d22be00a2bce"
precision_score(y_true=y_test, y_pred=pred)

# + id="qhQ9u77e_55h" outputId="a50a92c6-6d6a-4436-c70f-ea904e41380f"
recall_score(y_true=y_test, y_pred=pred)

# + [markdown] id="aK0E93HM_55h"
# # Optimize

# + id="BOkMWRIt_55h" outputId="0c0987cb-1036-4077-9fe8-0054a78fa69a"
XGBClassifier()

# + id="h0sBkWEu_55i" outputId="f8f1adc4-3656-491b-d193-5158d69e3387"
# kaggle에서 최적화 모델 (max_depth=5 and min_child_weight=1)

param_grid={
    'clf__n_estimators' : [10, 50, 100, 200, 400, 600],
    'clf__learning_rate' : [0.1, 0.2, 0.3, 0.4]
}
grid=GridSearchCV(pipe,
             param_grid,
             scoring='accuracy',
             n_jobs=-1,
             cv=10,
             verbose=3)
grid.fit(X_train, y_train)

# + id="bBpw7Btd_55i" outputId="4170d3b0-ff20-4b01-fc6c-8a6b975b35a3"
temp=pd.DataFrame(grid.cv_results_)[['param_clf__learning_rate', 'param_clf__n_estimators', 'mean_test_score']]
temp

# + id="SwQr9FAK_55i"
x=np.array(temp['param_clf__learning_rate']).reshape(6,4)
y=np.array(temp['param_clf__n_estimators']).reshape(6,4)
z=np.array(temp['mean_test_score']).reshape(6,4)

# + id="7fU1evVM_55j" outputId="df5bbd14-0398-43f8-ed9a-9144b72bf19f"
cntr = plt.contour(x, y, z, levels=np.arange(0.66, 0.70, 0.005))
plt.clabel(cntr, inline=2, fontsize=10)
plt.colorbar()
plt.show()

# + id="YSKtNb39_55j" outputId="0c9b5daa-a150-422c-e9f7-4eadf33103fd"
XGBClassifier()

# + id="CxyD9EnA_55j" outputId="a8ab41d7-3b81-44b3-d560-7c2dad5ad636"
# kaggle에서 최적화 모델 (max_depth=5 and min_child_weight=1)

param_grid={
    'clf__n_estimators' : [500, 700, 900],
    'clf__learning_rate' : [0.1],
    'clf__max_depth':[5, 7, 9, 12],
    'clf__min_child_weight':[1, 3, 5]
}
grid=GridSearchCV(pipe,
             param_grid,
             scoring='accuracy',
             n_jobs=-1,
             cv=10,
             verbose=3)
grid.fit(X_train, y_train)

# + [markdown] id="vtYDr9Ml_55k"
# # ROC

# + id="aElgFnnv_55k"
pd.DataFrame(grid.cv_results_).to_csv('grid.csv')

# + id="QszDSLXX_55k" outputId="056fa4d7-8e3d-4df5-bfdc-cbaa48367ee0"
fig, ax_roc = plt.subplots(figsize=(6, 5))
plot_roc_curve(pipe, X_test, y_test, ax=ax_roc)
plt.title('ROC Curve', fontsize=15)

# + [markdown] id="G-dmqYB4_55k"
# # Learning Curve

# +
from sklearn.model_selection import ShuffleSplit, learning_curve


def plot_learning_curve(estimator,
                        title,
                        X, y,
                        axes=None,
                        ylim=None,
                        cv=None,
                        n_jobs=None,
                        train_sizes=np.linspace(.1, 1.0, 5)):

    if axes is None:
        _, axes = plt.subplots(figsize=(6, 5))

    axes.set_title(title)
    if ylim is not None:
        axes.set_ylim(*ylim)
    axes.set_xlabel("Training examples")
    axes.set_ylabel("Score")

    train_sizes, train_scores, test_scores, fit_times, _ = \
        learning_curve(estimator, X, y, cv=cv, n_jobs=n_jobs,
                       train_sizes=train_sizes,
                       return_times=True)
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    fit_times_mean = np.mean(fit_times, axis=1)
    fit_times_std = np.std(fit_times, axis=1)

    # Plot learning curve
    axes.grid()
    axes.fill_between(train_sizes, train_scores_mean - train_scores_std,
                         train_scores_mean + train_scores_std, alpha=0.1,
                         color="r")
    axes.fill_between(train_sizes, test_scores_mean - test_scores_std,
                         test_scores_mean + test_scores_std, alpha=0.1,
                         color="g")
    axes.plot(train_sizes, train_scores_mean, 'o-', color="r",
                 label="Training score")
    axes.plot(train_sizes, test_scores_mean, 'o-', color="g",
                 label="Cross-validation score")
    axes.legend(loc="best")

    return plt

title = "Learning Curves (XGBoost)"

# Pipeline
ordinal_cols=['cellularity']
standard_cols=['age_at_diagnosis', 'lymph_nodes_examined_positive','mutation_count','nottingham_prognostic_index','tumor_size']
onehot_cols=['type_of_breast_surgery','cancer_type_detailed','pam50_+_claudin-low_subtype','cohort', 'er_status_measured_by_ihc','er_status', 'her2_status_measured_by_snp6','her2_status','tumor_other_histologic_subtype','inferred_menopausal_state','integrative_cluster', 'oncotree_code', 'pr_status']

onehot=Pipeline([
    ('oridinal', OrdinalEncoder(handle_unknown='use_encoded_value',
                                unknown_value=np.nan)),
    ('onehot', OneHotEncoder())
])
column_trans=ColumnTransformer([
    ('stadardard', StandardScaler(), standard_cols),
    ('ordinal', OrdinalEncoder(categories=[['Low','Moderate','High']], handle_unknown='use_encoded_value', unknown_value=np.nan), ordinal_cols),
    ('onehot', onehot, onehot_cols)],
    remainder="passthrough"
)

pipe=Pipeline([
    ('preprocess', column_trans),
    ('clf', XGBClassifier(random_state=1234,
                          eval_metric='logloss',
                          objective='binary:logistic',
                          n_estimators=100,
                          max_depth=8,
                          min_child_weight=3,
                          learning_rate=0.0001,
                          colsample_bylevel=0.9,
                          colsample_bytree=0.4,
                          gamma=0))])
cv = ShuffleSplit(n_splits=10, test_size=0.2, random_state=0)

plot_learning_curve(pipe, title, X, y, cv=cv, n_jobs=-1)
plt.show()

# + id="YOcL-qBb_55l"
# learning curve 그릴 때, sklearn version 0.24.2 필수
