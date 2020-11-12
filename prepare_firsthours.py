import pandas as pd
import numpy as np
from collections import Counter
from datetime import datetime
import os

pd.set_option('display.max_columns', None)

def compute_age(row):
    t1 = datetime.strptime(row['ADMITTIME'], '%Y-%m-%d %H:%M:%S')
    t2 = datetime.strptime(row['DOB'], '%Y-%m-%d %H:%M:%S')
    t = (t1 - t2).days//30
    return t

def compute_gender(row):
    if row['GENDER'] == 'F':
        return 0
    elif row['GENDER'] == 'M':
        return 1
    else:
        return -1

def is_first_hours(row):
    t1 = datetime.strptime(row['ADMITTIME'], '%Y-%m-%d %H:%M:%S')
    t2 = datetime.strptime(row['CHARTTIME'], '%Y-%m-%d %H:%M:%S')
    t = (t2 - t1).total_seconds()
    if t < hours*3600:
        return 1
    else:
        return 0

if __name__ == "__main__":

    hours = 48
    data_path = 'raw_data'

    # basic info
    df_admissions = pd.read_csv(os.path.join(data_path, 'ADMISSIONS.csv')) # diagnosis before hospital, gender, age
    df_icustays = pd.read_csv(os.path.join(data_path, 'ICUSTAYS.csv'))
    df_patients = pd.read_csv(os.path.join(data_path, 'PATIENTS.csv'))
    df_diagnosis_icd = pd.read_csv(os.path.join(data_path, 'DIAGNOSES_ICD.csv')) # diagnosis after hospital

    # dicts
    df_d_icd_diagnosis = pd.read_csv(os.path.join(data_path, 'D_ICD_DIAGNOSES.csv'))
    df_d_items = pd.read_csv(os.path.join(data_path, 'D_ITEMS.csv'))
    df_d_labitems = pd.read_csv(os.path.join(data_path, 'D_LABITEMS.csv'))

    # ts_bed
    df_chartevents = pd.read_csv(os.path.join(data_path, 'CHARTEVENTS.csv')) # yes, 19 distinct items
    df_labevents = pd.read_csv(os.path.join(data_path, 'LABEVENTS.csv')) # yes, 821 distinct items
    df_outputevents = pd.read_csv(os.path.join(data_path, 'OUTPUTEVENTS.csv')) # no
    df_emr_symptoms = pd.read_csv(os.path.join(data_path, 'EMR_SYMPTOMS.csv')) # no
    df_microbiologyevents = pd.read_csv(os.path.join(data_path, 'MICROBIOLOGYEVENTS.csv')) # no
    df_or_exam_reports = pd.read_csv(os.path.join(data_path, 'OR_EXAM_REPORTS.csv')) # no
    df_prescriptions = pd.read_csv(os.path.join(data_path, 'PRESCRIPTIONS.csv')) # no

    # ts_operation
    df_surgery_vital_signs = pd.read_csv(os.path.join(data_path, 'SURGERY_VITAL_SIGNS.csv')) # no

    # demographic
    df_demo = df_admissions.merge(df_patients, left_on='SUBJECT_ID', right_on='SUBJECT_ID', how='left')
    df_demo['age_month'] = df_demo.apply(lambda row: compute_age(row), axis=1)
    df_demo['gender_is_male'] = df_demo.apply(lambda row: compute_gender(row), axis=1)
    df_weight = df_chartevents[df_chartevents['ITEMID']==1014]
    df_weight = df_weight.groupby(['HADM_ID']).first().reset_index()
    df_demo = df_demo.merge(df_weight, left_on='HADM_ID', right_on='HADM_ID', how='left')
    df_demo['weight_kg'] = df_demo['VALUENUM']
    df_demo['SUBJECT_ID'] = df_demo['SUBJECT_ID_x']
    cols = ['SUBJECT_ID', 'HADM_ID', 'ADMITTIME', 'age_month', 'gender_is_male', 'weight_kg', 'HOSPITAL_EXPIRE_FLAG']
    df_demo = df_demo[cols]

    # chartevents
    # df_chartevents_small = df_chartevents.iloc[:1000]
    df_chartevents_firsthours = df_chartevents.merge(df_demo, left_on='HADM_ID', right_on='HADM_ID', how='left')
    df_chartevents_firsthours['is_first_hours'] = df_chartevents_firsthours.apply(lambda row: is_first_hours(row), axis=1)
    df_chartevents_firsthours = df_chartevents_firsthours[df_chartevents_firsthours.is_first_hours==1]
    df_chartevents_firsthours['SUBJECT_ID'] = df_chartevents_firsthours['SUBJECT_ID_x']
    cols = ['HADM_ID', 'ITEMID', 'VALUENUM']
    df_chartevents_firsthours = df_chartevents_firsthours[cols].reset_index(drop=True)

    df_chartevents_firsthours_max = df_chartevents_firsthours.groupby(['HADM_ID', 'ITEMID']).max().reset_index()
    df_chartevents_firsthours_max = df_chartevents_firsthours_max.pivot_table(index='HADM_ID', columns='ITEMID', values='VALUENUM')
    df_chartevents_firsthours_max = df_chartevents_firsthours_max.reset_index()
    df_chartevents_firsthours_max.columns = ['chart_{}_max'.format(i) for i in list(df_chartevents_firsthours_max.columns)]

    df_chartevents_firsthours_min = df_chartevents_firsthours.groupby(['HADM_ID', 'ITEMID']).min().reset_index()
    df_chartevents_firsthours_min = df_chartevents_firsthours_min.pivot_table(index='HADM_ID', columns='ITEMID', values='VALUENUM')
    df_chartevents_firsthours_min = df_chartevents_firsthours_min.reset_index()
    df_chartevents_firsthours_min.columns = ['chart_{}_min'.format(i) for i in list(df_chartevents_firsthours_min.columns)]

    # labevents
    # df_labevents_small = df_labevents.iloc[:1000]
    df_labevents_firsthours = df_labevents.merge(df_demo, left_on='HADM_ID', right_on='HADM_ID', how='left')
    df_labevents_firsthours['is_first_hours'] = df_labevents_firsthours.apply(lambda row: is_first_hours(row), axis=1)
    df_labevents_firsthours = df_labevents_firsthours[df_labevents_firsthours.is_first_hours==1]
    df_labevents_firsthours['SUBJECT_ID'] = df_labevents_firsthours['SUBJECT_ID_x']
    cols = ['HADM_ID', 'ITEMID', 'VALUENUM']
    df_labevents_firsthours = df_labevents_firsthours[cols].reset_index(drop=True)

    df_labevents_firsthours_max = df_labevents_firsthours.groupby(['HADM_ID', 'ITEMID']).max().reset_index()
    df_labevents_firsthours_max = df_labevents_firsthours_max.pivot_table(index='HADM_ID', columns='ITEMID', values='VALUENUM')
    df_labevents_firsthours_max = df_labevents_firsthours_max.reset_index()
    df_labevents_firsthours_max.columns = ['lab_{}_max'.format(i) for i in list(df_labevents_firsthours_max.columns)]

    df_labevents_firsthours_min = df_labevents_firsthours.groupby(['HADM_ID', 'ITEMID']).min().reset_index()
    df_labevents_firsthours_min = df_labevents_firsthours_min.pivot_table(index='HADM_ID', columns='ITEMID', values='VALUENUM')
    df_labevents_firsthours_min = df_labevents_firsthours_min.reset_index()
    df_labevents_firsthours_min.columns = ['lab_{}_min'.format(i) for i in list(df_labevents_firsthours_min.columns)]

    # df all
    df_all = df_demo.merge(df_chartevents_firsthours_max, left_on='HADM_ID', right_on='chart_HADM_ID_max', how='left')
    df_all = df_all.merge(df_chartevents_firsthours_min, left_on='HADM_ID', right_on='chart_HADM_ID_min', how='left')
    df_all = df_all.merge(df_labevents_firsthours_max, left_on='HADM_ID', right_on='lab_HADM_ID_max', how='left')
    df_all = df_all.merge(df_labevents_firsthours_min, left_on='HADM_ID', right_on='lab_HADM_ID_min', how='left')
    cols = list(df_all.columns)
    remove_cols = ['chart_HADM_ID_max', 'chart_HADM_ID_min', 'lab_HADM_ID_max', 'lab_HADM_ID_min']
    cols = [x for x in cols if x not in remove_cols]
    df_all = df_all[cols]
    df_all.to_csv('icu_first{}hours.csv'.format(hours), index=False)

