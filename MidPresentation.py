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

import sklearn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# + id="E7edO85JAbZm" outputId="1d1f3846-0a29-4b00-bf37-ac30dd938d85"
df=pd.read_csv('METABRIC_RNA_Mutation.csv')
df

# + id="L4OGejc3AbZn" outputId="fdcd8286-0568-492a-ab40-b18def9955b7"
len(df.columns)

# + id="gbX4hOyyAbZo" outputId="ec73c777-d122-4ca3-da2e-7ee6623633f9"
list(df.columns)

# + id="Z2Ktg_gmAbZo" outputId="da253697-e8fa-4b20-8b80-67b891b59f2d"
df['cancer_type_detailed']

# + [markdown] id="NGJr44xcAbZp"
# ## 수술 종류와 테라피의 조합에 따른 NPI

# + id="o8fcMfZUAbZp"
# NPI를 보는 것 이므로 surgery 필수 -> surgery  NaN은 drop
# surgery 별 NPI chk
# surgery에서 therapyh 종류 별 (radiotheraphy, chemotheraphy, hormonetheraphys) 

# + id="L0LQ3UjdAbZp"
npi=df.dropna(subset=['type_of_breast_surgery']).copy()

# + id="t8GC1A8SAbZq" outputId="b4542325-479f-42c0-f38f-58d3863eb72c"
npi['type_of_breast_surgery'].value_counts()

# + id="Fr4ne8HIAbZq" outputId="7fe4bde3-1bc9-473d-a359-28fa3d826b07"
plt.figure(figsize=(14,4))
m=npi.loc[list(npi['type_of_breast_surgery']=='MASTECTOMY'),['nottingham_prognostic_index']]
b=npi.loc[list(npi['type_of_breast_surgery']=='BREAST CONSERVING'),['nottingham_prognostic_index']]
plt.xticks(np.arange(1,7,0.25), rotation=45)
plt.xlabel('NPI (Nottingham_Prognostic_Index)')
plt.ylabel('Number of Patients')
plt.hist(m, align='mid', bins=np.arange(1,6.25,0.05))
plt.hist(b, align='mid', bins=np.arange(1,6.25,0.05))
plt.legend(['MASTECTOMY', 'BREAST CONSERVING'])
plt.title('NPI distribution by type of Surgery', fontsize=15)
plt.show()

# + id="OJ6tVUyRAbZr" outputId="b443d1ef-6c8a-4861-91a6-fe6fa5445562"
npi1=df.dropna(subset=['chemotherapy','hormone_therapy','radio_therapy','nottingham_prognostic_index']).copy()
npi1

# + id="pLIh0H1xAbZr" outputId="5ae00791-a07d-42a7-8722-7eeb3c57f877"
m=npi1.loc[list(npi1['type_of_breast_surgery']=='MASTECTOMY'),['chemotherapy','hormone_therapy','radio_therapy','nottingham_prognostic_index']]
b=npi1.loc[list(npi1['type_of_breast_surgery']=='BREAST CONSERVING'),['chemotherapy','hormone_therapy','radio_therapy','nottingham_prognostic_index']]
b

# + id="pMtPX3ClAbZr" outputId="3ec9e75d-b2d4-4a1b-d70c-8888128c3405"
plt.figure(figsize=(30,8))
a=m.loc[[idx for idx in m.index if m.loc[idx, 'chemotherapy']==0 and m.loc[idx, 'hormone_therapy']==1 and m.loc[idx, 'radio_therapy']==0 ],'nottingham_prognostic_index']
b=m.loc[[idx for idx in m.index if m.loc[idx, 'chemotherapy']==0 and m.loc[idx, 'hormone_therapy']==0 and m.loc[idx, 'radio_therapy']==0 ],'nottingham_prognostic_index']
c=m.loc[[idx for idx in m.index if m.loc[idx, 'chemotherapy']==0 and m.loc[idx, 'hormone_therapy']==1 and m.loc[idx, 'radio_therapy']==1 ],'nottingham_prognostic_index']
d=m.loc[[idx for idx in m.index if m.loc[idx, 'chemotherapy']==1 and m.loc[idx, 'hormone_therapy']==0 and m.loc[idx, 'radio_therapy']==1 ],'nottingham_prognostic_index']
e=m.loc[[idx for idx in m.index if m.loc[idx, 'chemotherapy']==1 and m.loc[idx, 'hormone_therapy']==1 and m.loc[idx, 'radio_therapy']==1 ],'nottingham_prognostic_index']
f=m.loc[[idx for idx in m.index if m.loc[idx, 'chemotherapy']==1 and m.loc[idx, 'hormone_therapy']==0 and m.loc[idx, 'radio_therapy']==0 ],'nottingham_prognostic_index']
g=m.loc[[idx for idx in m.index if m.loc[idx, 'chemotherapy']==0 and m.loc[idx, 'hormone_therapy']==0 and m.loc[idx, 'radio_therapy']==1 ],'nottingham_prognostic_index']
h=m.loc[[idx for idx in m.index if m.loc[idx, 'chemotherapy']==1 and m.loc[idx, 'hormone_therapy']==1 and m.loc[idx, 'radio_therapy']==0 ],'nottingham_prognostic_index']

plt.xlim(1,6.5)
plt.boxplot([a,b,c,d,e,f,g,h],vert=False,labels=['(0,1,0)','(0,0,0)','(0,1,1)','(1,0,1)','(1,1,1)','(1,0,0)','(0,0,1)','(1,1,0)'])
plt.xlabel('NPI (Nottingham_Prognostic_Index)')
plt.ylabel('Combination of Therapy')

plt.show()
# -

b=npi1.loc[list(npi1['type_of_breast_surgery']=='BREAST CONSERVING'),['chemotherapy','hormone_therapy','radio_therapy','nottingham_prognostic_index']]

# + id="EVDqAcSuAbZs" outputId="085b8597-0ec8-4f17-fd9b-1b22b8ea85de"
plt.figure(figsize=(30,8))
a=b.loc[[idx for idx in b.index if b.loc[idx, 'chemotherapy']==0 and b.loc[idx, 'hormone_therapy']==1 and b.loc[idx, 'radio_therapy']==0 ],'nottingham_prognostic_index']
B=b.loc[[idx for idx in b.index if b.loc[idx, 'chemotherapy']==0 and b.loc[idx, 'hormone_therapy']==0 and b.loc[idx, 'radio_therapy']==0 ],'nottingham_prognostic_index']
c=b.loc[[idx for idx in b.index if b.loc[idx, 'chemotherapy']==0 and b.loc[idx, 'hormone_therapy']==1 and b.loc[idx, 'radio_therapy']==1 ],'nottingham_prognostic_index']
d=b.loc[[idx for idx in b.index if b.loc[idx, 'chemotherapy']==1 and b.loc[idx, 'hormone_therapy']==0 and b.loc[idx, 'radio_therapy']==1 ],'nottingham_prognostic_index']
e=b.loc[[idx for idx in b.index if b.loc[idx, 'chemotherapy']==1 and b.loc[idx, 'hormone_therapy']==1 and b.loc[idx, 'radio_therapy']==1 ],'nottingham_prognostic_index']
f=b.loc[[idx for idx in b.index if b.loc[idx, 'chemotherapy']==1 and b.loc[idx, 'hormone_therapy']==0 and b.loc[idx, 'radio_therapy']==0 ],'nottingham_prognostic_index']
g=b.loc[[idx for idx in b.index if b.loc[idx, 'chemotherapy']==0 and b.loc[idx, 'hormone_therapy']==0 and b.loc[idx, 'radio_therapy']==1 ],'nottingham_prognostic_index']
h=b.loc[[idx for idx in b.index if b.loc[idx, 'chemotherapy']==1 and b.loc[idx, 'hormone_therapy']==1 and b.loc[idx, 'radio_therapy']==0 ],'nottingham_prognostic_index']

plt.xlim(1,6.5)
plt.boxplot([a,B,c,d,e,f,g,h],vert=False,labels=['(0,1,0)','(0,0,0)','(0,1,1)','(1,0,1)','(1,1,1)','(1,0,0)','(0,0,1)','(1,1,0)'])
plt.xlabel('NPI (Nottingham_Prognostic_Index)')
plt.ylabel('Combination of Therapy')

plt.show()

# + id="IcchJlBgAbZs" outputId="603327c2-1a69-47c4-a8a2-c916c2951032"
m[['chemotherapy','hormone_therapy','radio_therapy']].value_counts()

# + id="KxnXCgi9AbZt" outputId="88b6564b-93fb-4373-d1d0-bdaf07259159"
b[['chemotherapy','hormone_therapy','radio_therapy']].value_counts()

# + [markdown] id="dNY9oqGAAbZt"
# ## 병리학적인 분석과 현미경 분석의 연관성

# + id="dFGQULSbAbZu"
# tumor_other_histologic_subtype
# neoplasm_histologic_grade
# 두 변수의 관계

# + id="Fek9qLwkAbZu" outputId="bf4e6f45-218f-4f12-9989-8caf3e70b867"
df[['tumor_other_histologic_subtype']].value_counts()

# + id="2RV_zvy9AbZu" outputId="4b6935e8-0a01-49e4-ee26-2e3455ffffe2"
df[['neoplasm_histologic_grade']].value_counts()

# + id="4Yo-TlUFAbZv" outputId="482893f0-3c7d-4cee-9c35-cd169f50bf4c"
corr=df.dropna(subset=['neoplasm_histologic_grade', 'tumor_other_histologic_subtype'])[['neoplasm_histologic_grade', 'tumor_other_histologic_subtype']]
corr

# + id="d_L3FfPWAbZv" outputId="9dc8c086-35b5-4d04-db98-a2729e26bee9"
a=corr[corr['neoplasm_histologic_grade']==3.0].groupby('tumor_other_histologic_subtype').count()
b=corr[corr['neoplasm_histologic_grade']==2.0].groupby('tumor_other_histologic_subtype').count()
c=corr[corr['neoplasm_histologic_grade']==1.0].groupby('tumor_other_histologic_subtype').count()
m1=pd.merge(a, b, on="tumor_other_histologic_subtype", how="outer")
m2=pd.merge(m1, c, on="tumor_other_histologic_subtype", how="outer")
m2.fillna(0, inplace=True)
m2.columns=['NHG_3','NHG_2','NHG_1']
m2

# + id="DuMVH4QsAbZv" outputId="f4239063-45a7-4a63-cc89-8ab1808672f6"
plt.figure(figsize=(10,4))
plt.barh(m2.index,m2['NHG_1'],color='lightsteelblue')
plt.barh(m2.index,m2['NHG_2'],left=m2['NHG_1'],color='cornflowerblue')
plt.barh(m2.index,m2['NHG_3'],left=m2['NHG_2'],color='royalblue')
plt.legend(['NHG_1','NHG_2','NHG_3'])
plt.ylabel('Histologic Subtype')
plt.xlabel('the Number of each NHG Class')
plt.show()

# + [markdown] id="xt8YIvBEAbZv"
# ## stage가 같은 암 환자들의 테라피에 따른 생존기간

# + id="3C5d8-uQAbZw" outputId="8aa2fda8-9e1b-41e7-fb7c-2698d29d0a31"
eff=df.dropna(subset=['tumor_stage','overall_survival_months', 'chemotherapy', 'radio_therapy', 'hormone_therapy'])[['tumor_stage','overall_survival_months', 'chemotherapy', 'radio_therapy', 'hormone_therapy']].copy()
eff['isTherapy']=[1 if (eff.loc[idx, 'chemotherapy']+eff.loc[idx, 'radio_therapy']+eff.loc[idx, 'hormone_therapy'])>0 else 0 for idx in eff.index]
eff

# + id="U8BVaEA2AbZw" outputId="fe60a844-2f11-4f6e-f5d0-53b241422be2"
eff['tumor_stage'].value_counts()

# + id="VwQMBO46AbZw" outputId="980e3d48-91ac-4a25-bbe0-a288206d67ac"
plt.figure(figsize=(4,10))
stage_0=eff[eff['tumor_stage']==0]
stage_0_no=stage_0[stage_0['isTherapy']==0]
stage_0_yes=stage_0[stage_0['isTherapy']==1]
plt.boxplot([stage_0_yes['overall_survival_months'], stage_0_no['overall_survival_months']], labels=['Yes','No'])
plt.xlabel('Is Therapy')
plt.ylim(0,400)
plt.ylabel('Overall Survival Months')
plt.title('Tumor Stage 0 Patients', fontsize=15)
plt.show()

# + id="Z8DX5GpKAbZw" outputId="e2d89687-0d6b-4f44-b045-cd0b1721e444"
plt.figure(figsize=(4,10))
stage_1=eff[eff['tumor_stage']==1]
stage_1_no=stage_1[stage_1['isTherapy']==0]
stage_1_yes=stage_1[stage_1['isTherapy']==1]
plt.boxplot([stage_1_yes['overall_survival_months'], stage_1_no['overall_survival_months']], labels=['Yes','No'])
plt.xlabel('Is Therapy')
plt.ylim(0,400)
plt.ylabel('Overall Survival Months')
plt.title('Tumor Stage 1 Patients', fontsize=15)
plt.show()

# + id="UzLqh2PcAbZw" outputId="c288c106-2799-4700-a050-b96fead2d4bf"
plt.figure(figsize=(4,10))
stage_2=eff[eff['tumor_stage']==2]
stage_2_no=stage_2[stage_2['isTherapy']==0]
stage_2_yes=stage_2[stage_2['isTherapy']==1]
plt.boxplot([stage_2_yes['overall_survival_months'], stage_2_no['overall_survival_months']], labels=['Yes','No'])
plt.xlabel('Is Therapy')
plt.ylim(0,400)
plt.ylabel('Overall Survival Months')
plt.title('Tumor Stage 2 Patients', fontsize=15)
plt.show()

# + id="aiMD5-oxAbZx" outputId="59b0b804-9eec-4b89-86fe-0baab510e2a2"
plt.figure(figsize=(4,10))
stage_3=eff[eff['tumor_stage']==3]
stage_3_no=stage_3[stage_3['isTherapy']==0]
stage_3_yes=stage_3[stage_3['isTherapy']==1]
plt.boxplot([stage_3_yes['overall_survival_months'], stage_3_no['overall_survival_months']], labels=['Yes','No'])
plt.xlabel('Is Therapy')
plt.ylim(0,400)
plt.ylabel('Overall Survival Months')
plt.title('Tumor Stage 3 Patients', fontsize=15)
plt.show()

# + id="AE2FngdAAbZx" outputId="5f096bc6-cfeb-48bb-db27-5d4e1e3f7fb0"
plt.figure(figsize=(4,10))
stage_4=eff[eff['tumor_stage']==4]
stage_4_no=stage_4[stage_4['isTherapy']==0]
stage_4_yes=stage_4[stage_4['isTherapy']==1]
plt.boxplot([stage_4_yes['overall_survival_months'], stage_4_no['overall_survival_months']], labels=['Yes','No'])
plt.xlabel('Is Therapy')
plt.ylim(0,400)
plt.ylabel('Overall Survival Months')
plt.title('Tumor Stage 4 Patients', fontsize=15)
plt.show()

# + id="piano-listing" outputId="adff0d41-4c68-4c93-ef76-6b851181abab"
data = pd.read_csv('METABRIC_RNA_Mutation.csv')
df = data.iloc[:, :31]

# + [markdown] id="experimental-generation"
# ## 암의 종류에 따른 암세포성의 차이

# + id="express-eagle"
df1 = df[["cancer_type_detailed", "cellularity"]]
df1 = df1.dropna(axis = 0)

type_dict = {"Breast Invasive Ductal Carcinoma":"IDC",
             "Breast Invasive Lobular Carcinoma":"ILC",
             "Breast Invasive Mixed Mucinous Carcinoma":"IMMC",
             "Breast Mixed Ductal and Lobular Carcinoma":"MDLC"}

df1 = df1.replace({"cancer_type_detailed":type_dict})

# + id="threaded-consultancy"
df1_l = df1[df1['cellularity'] == "Low"]
df1_m = df1[df1['cellularity'] == "Moderate"]
df1_h = df1[df1['cellularity'] == "High"]

df1_l = df1_l.groupby(["cancer_type_detailed"], as_index = False).count()
df1_m = df1_m.groupby(["cancer_type_detailed"], as_index = False).count()
df1_h = df1_h.groupby(["cancer_type_detailed"], as_index = False).count()

df1_m = df1_m.drop([df1.index[4]]) #데이터 내 1명 존재하는 암 종류 삭제

# + id="psychological-paper" outputId="2f5ff0cd-76d6-4944-d317-a6ffc2193d51"
x = df1_l['cancer_type_detailed'].values.tolist()
y1 = df1_l['cellularity'].values.tolist()
y2 = df1_m['cellularity'].values.tolist()
y3 = df1_h['cellularity'].values.tolist()

dic = {'cancer type':x, 'Low':y1, 'Moderate':y2, 'High':y3}
data = pd.DataFrame(dic, index = np.arange(5))
data

# + id="radical-exception" outputId="262b9d1b-9e7e-4c50-868a-b0a1bede6520"
data[['Low', 'Moderate', 'High']].plot(kind="bar", figsize=(10,6))
plt.title('Number of Cellularity by Cancer Type', fontsize = 15, fontweight='bold')
plt.xticks([0, 1, 2, 3, 4], labels=["Breast", "IDC", "ILC", "IMMC", "MDLC"])
plt.xticks(rotation=0)

plt.legend(title='Cellularity')
plt.xlabel('cancer type')
plt.ylabel('number of patients')

# + id="ready-incentive"
cell_value = {"Low":5, "Moderate":20, "High":35}
#Low cellularity was defined as 10 or fewer cell clusters,
#moderate cellularity was defined as 11‐30 clusters,
#and high cellularity was defined as more than 30 clusters.
#-> 중간값으로 대체

df1 = df1.replace({"cellularity":cell_value})

# + id="senior-bangkok" outputId="42122302-810c-49ee-82d1-d75b4a511add"
df1 = df1.groupby(["cancer_type_detailed"], as_index = False).mean()
df1 = df1.drop([df1.index[5]]) #데이터 내 1명 존재하는 암 종류 삭제
df1

# + id="enhanced-mills" outputId="ea304617-fd89-4add-8c3c-34b76b265db6"
plt.figure(figsize= (10, 6))
plt.title('Cellularity by Cancer Type', fontsize = 15, fontweight='bold')

plt.xlabel('cancer type')
plt.ylabel('cellularity')

plt.bar(df1['cancer_type_detailed'], df1['cellularity'], width = 0.5)
plt.plot(df1['cancer_type_detailed'], df1['cellularity'], color = 'red')

plt.show()

# + [markdown] id="scenic-aaron"
# ## 암의 단계에 따른 생존율

# + id="fewer-distributor"
df2 = df[['tumor_stage', 'death_from_cancer']]
df2 = df2.dropna(axis = 0)

# + id="designing-loading"
df2_liv = df2[df2['death_from_cancer'] == "Living"]
df2_dis = df2[df2['death_from_cancer'] == "Died of Disease"]
df2_oth = df2[df2['death_from_cancer'] == "Died of Other Causes"]

# + id="selected-practitioner"
df2_liv = df2_liv.groupby(["tumor_stage"], as_index = False).count()
df2_dis = df2_dis.groupby(["tumor_stage"], as_index = False).count()
df2_oth = df2_oth.groupby(["tumor_stage"], as_index = False).count()

# + id="homeless-lemon"
add = pd.DataFrame({'tumor_stage':0.0, 'death_from_cancer':0}, index = [0])
df2_dis = add.append(df2_dis, ignore_index=True)

add = pd.DataFrame({'tumor_stage':4.0, 'death_from_cancer':0}, index = [0])
df2_oth = df2_oth.append(add, ignore_index=True)

# + id="relative-thomas" outputId="c080588d-74bc-42cd-9cf3-8ffa2ed4fa47"
x = df2_liv['tumor_stage'].values.tolist()
y1 = df2_liv['death_from_cancer'].values.tolist()
y2 = df2_dis['death_from_cancer'].values.tolist()
y3 = df2_oth['death_from_cancer'].values.tolist()

dic = {'tumor stage':x, 'living':y1, 'died of disease':y2, 'died of others':y3}
data = pd.DataFrame(dic, index = np.arange(5))
data

# + id="musical-following" outputId="5700bf10-5b12-47f6-b891-a7fff46cda84"
data[['living', 'died of disease', 'died of others']].plot(kind='bar', figsize=(10,6), stacked=True)
plt.title('Patients\' State by Tumor Stage', fontsize = 15, fontweight='bold')
plt.xticks(rotation=0)

plt.legend(title='Patients\' State')
plt.xlabel('tumor stage')
plt.ylabel('number of patients')

# + id="protecting-darwin" outputId="40c4d7df-38d3-4741-f2e2-9904ee5c1ccc"
data['survivability'] = data['living'] / (data['living'] + data['died of disease'] + data['died of others'])
data

# + id="confirmed-logging" outputId="e0bf70d2-3f0f-4250-826f-1da51aba37cc"
data.info()

# + id="acknowledged-budget" outputId="1350518e-6b20-456d-cdcc-be339c77838a"
plt.figure(figsize= (10, 6))
plt.title('Survivability by Tumor Stage', fontsize = 15, fontweight='bold')

plt.xlabel('tumor stage')
plt.ylabel('survivability')

plt.bar(data['tumor stage'], data['survivability'], width = 0.5, color = '#86E57F')
plt.plot(data['tumor stage'], data['survivability'], color = '#F15F5F')

plt.show()

# + [markdown] id="greater-soviet"
# ## 종양 크기에 따른 생존율

# + id="royal-orleans" outputId="09bd24a2-34bf-4f1c-9c90-252b0b6d15d6"
plt.figure(figsize= (10, 6))
plt.title('State Distribution by Tumor Stage', fontsize = 15, fontweight='bold')

sns.distplot(df[df['death_from_cancer'] == "Living"]["tumor_size"],
             color="blue", label="Living", hist=False)

sns.distplot(df[df['death_from_cancer'] == "Died of Disease"]["tumor_size"],
             color="red", label="Died of Disease", hist=False)

sns.distplot(df[df['death_from_cancer'] == "Died of Other Causes"]["tumor_size"],
             color="green", label="Died of Other Causes", hist=False)

plt.legend(title="State")
plt.xlabel('tumor size')
plt.ylabel('density')

plt.show()

# + id="resistant-adult"
df3 = df[['tumor_size', 'death_from_cancer']]
df3 = df3.dropna(axis = 0)

death_dict = {"Died of Disease":0, "Died of Other Causes":0, "Living":1}
df3 = df3.replace({"death_from_cancer":death_dict})
df3.columns = ['tumor_size', 'survive']

# + id="inappropriate-arcade" outputId="a4ff0667-eed1-4207-9f36-a51c1c7a6178"
df3.info()

# + id="reflected-seventh"
df3_sum = df3.groupby(['tumor_size']).sum()
df3_cnt = df3.groupby(['tumor_size']).count()

x = df3_sum.index
y = []
sv = df3_sum[['survive']] / df3_cnt[['survive']]
for i in sv.values.tolist():
    y.append(i[0])

# + id="dirty-individual" outputId="11c327d9-fea5-48b7-ca12-c57318606547"
plt.figure(figsize= (10, 6))
plt.title('Scatter Plot about Survivability by Tumor Size', fontsize = 15, fontweight='bold')
sns.regplot(x=x, y=y, fit_reg=True)

plt.xlabel('tumor size')
plt.ylabel('survive/total')
plt.ylim([-0.1, 1.1])

plt.show()

# + [markdown] id="intense-tension"
#  

# + [markdown] id="complete-flour"
#  

# + [markdown] id="involved-catalyst"
#  

# + [markdown] id="scheduled-vanilla"
# ## 암의 단계에 따른 종양 크기 비교

# + id="comprehensive-orange" outputId="98648e4c-cbdf-4b55-ce39-ddd593abcae8"
color_dict = ["#9B59B6", "#3498DB", "#95A5A6", "#E74C3C", "#2ECC71"]

plt.figure(figsize= (15, 7))
sns.boxplot(x = "tumor_stage", y = "tumor_size", data = df, palette = color_dict)
plt.xlabel("timor stage")
plt.ylabel("tumor size")
plt.title("Tumor Size by Tumor Stage", fontsize = 15, fontweight='bold')
plt.show()
