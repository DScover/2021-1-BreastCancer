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

# +
import pandas as pd
import numpy as np
import seaborn as sns
import time
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report,confusion_matrix
from sklearn.ensemble import RandomForestClassifier



# -

#csv파일을 pandas dataframe으로 Import
data = pd.read_csv('METABRIC_RNA_Mutation.csv')
data.shape #row, column 개수


#dataframe 3개만 확인
data.head(3)

#cancer_type_detailed,type_of_breast_surgery,cancer_type등의 정보는 영향이 없지 않을까 해서 data에서 날림. 문자열이라 날림.
X = data.drop(['cancer_type_detailed'], axis = 1)
X = X.drop(['type_of_breast_surgery'],axis=1)
X = X.drop(['cancer_type'],axis=1)
X = X.drop(['cellularity'],axis=1)
X = X.drop(['pam50_+_claudin-low_subtype'],axis=1)
X = X.drop(['her2_status'],axis=1)
X = X.drop(['inferred_menopausal_state'],axis=1)
X = X.drop(['integrative_cluster'],axis=1)
X = X.drop(['er_status'],axis=1)
X = X.drop(['her2_status_measured_by_snp6'],axis=1)
X = X.drop(['pr_status'],axis=1)
#결측값 제거 (없는 값 제거)
X = X.dropna(axis=1)

#정규표현식으로 알파벳 값 들어있는것 제거 시도,,,, 실패
#정규표현식을 배울 수 있음.
#df["title"] = df["title"].str.replace(pat=r'[^\w]', repl=r'', regex=True)
#이런식으로 쓰면 특수문자를 날리라는 의미임.
#regex를 해줘야 정규식 반영이 됨.
X = X.replace({'A-Z'}, regex=True)

#혹시나 모르는 결측 값 제거
X = X.dropna(axis=1)

#for loop로 제거 시도 but 실패
for i in X:
    if X[i].dtype=='object':
        count=0
        for j in X[i]:
            if type(j)=='str':
                if len(j)>1:
                    X[i].replace([j], '1')
                count+=1

#모든 Contents를 숫자로 전부 표현 했을 때 숫자가 아닌 값들에 대해서 1로 채우라는 의미
X=X.apply(pd.to_numeric,errors='coerce').fillna(1)

#foxo3_mut,ncor1_mut,tg_mut 등 비교적 앞쪽에 문자열이 있는 COLLUMN에 대해서 변화 확인.
X['ncor1_mut']

# input data에 대해서 model에 넣을 수 있게 numpy로 변형
x_input=X.to_numpy()
#input 정규화
x_train = StandardScaler().fit_transform(x_input)
#label로 쓰일 값 one hot vector로 생성
Y = pd.get_dummies(data['er_status_measured_by_ihc'])
#label 값 numpy로 변경
y_train=Y.to_numpy()
# tensorflow 나 pytorch로 작업하면 될듯함.


#dataset shape 확인
x_train.shape,y_train.shape
