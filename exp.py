import pandas as pd
import numpy as np
from collections import Counter, OrderedDict
from matplotlib import pyplot as plt
from tqdm import tqdm

from sklearn.model_selection import KFold, train_test_split
from sklearn.linear_model import LogisticRegression as LR
from sklearn.ensemble import RandomForestClassifier as RF
from sklearn.metrics import classification_report
from sklearn.metrics import roc_curve, roc_auc_score

from baseline_prism_iii import prism_iii

import warnings
warnings.filterwarnings('ignore') 
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)

plt.rcParams['pdf.fonttype'] = 42

def my_eval(gt, y_pred_proba):
    """
    y_pred_proba are float
    gt, y_pred are binary
    """
    
    ret = OrderedDict({})
    ret['auroc'] = roc_auc_score(gt, y_pred_proba)
    ret['fpr'], ret['tpr'], _ = roc_curve(gt, y_pred_proba)    

    return ret

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def print_lr_model(m, x_cols):
    
    final_m_coef_ = m.coef_[0]
    final_m_intercept_ = m.intercept_[0]
    model_str = ''
    for i in range(len(x_cols)):
        model_str += '{:.4f}*{} + '.format(final_m_coef_[i], x_cols[i])
    model_str += '{:.4f}'.format(final_m_intercept_)
    model_str = 'sigmoid(' + model_str + ')'
    print('Final model: Probability =', model_str)
    
    
if __name__ == "__main__":    
    
    seed = 0
    n_fold = 10
    max_n_features = 64
    max_topK = 32
    
    # ------------------------ read data ------------------------
    np.random.seed(seed)
    df = pd.read_csv('icu_first24hours.csv')
    
    MAX_MISSING_RATE = 1.0 # use all features
    df_missing_rate = df.isnull().mean().sort_values().reset_index()
    df_missing_rate.columns = ['col','missing_rate']
    cols = list(df_missing_rate[df_missing_rate['missing_rate'] < MAX_MISSING_RATE].col.values)
    df = df[cols]
    
    shuffle_idx = np.random.permutation(df.shape[0])
    split_idx = int(0.8*df.shape[0])
    train_idx = shuffle_idx[:split_idx]
    test_idx = shuffle_idx[split_idx:]
    print('data split check value should be {} (72598077)'.format(np.sum(train_idx)))
    df_test = df.iloc[test_idx]
    df = df.iloc[train_idx]
    print('train/val set: ', df.shape, '; test set: ', df_test.shape)
    
    x_cols_all = ['age_month', 'gender_is_male'] + cols[6:]
    X = np.nan_to_num(df[x_cols_all].values)
    X_test = np.nan_to_num(df_test[x_cols_all].values)
    y = df['HOSPITAL_EXPIRE_FLAG'].values
    y_test = df_test['HOSPITAL_EXPIRE_FLAG'].values
    
    # ------------------------ demo stat ------------------------
    print(np.mean(df.age_month))
    print(np.std(df.age_month))
    print(Counter(df.gender_is_male), Counter(df.gender_is_male)[0]/df.shape[0])
    print(Counter(df.HOSPITAL_EXPIRE_FLAG), Counter(df.HOSPITAL_EXPIRE_FLAG)[1]/df.shape[0])
    
    print(np.mean(df_test.age_month))
    print(np.std(df_test.age_month))
    print(Counter(df_test.gender_is_male), Counter(df_test.gender_is_male)[0]/df_test.shape[0])
    print(Counter(df_test.HOSPITAL_EXPIRE_FLAG), Counter(df_test.HOSPITAL_EXPIRE_FLAG)[1]/df_test.shape[0])

    age1 = df.age_month
    age2 = df_test.age_month
    fig, ax = plt.subplots(2,1, figsize=(6,4))
    ax[0].hist(age1, bins=100, color='tab:blue')
    ax[0].legend(['Development Set'], fontsize=12)
    ax[0].set_xlabel('Age (month)', fontsize=16)
    ax[0].set_ylabel('Count', fontsize=16)
    ax[1].hist(age2, bins=100, color='tab:red')
    ax[1].legend(['Test Set'], fontsize=12)
    ax[1].set_xlabel('Age (month)', fontsize=16)
    ax[1].set_ylabel('Count', fontsize=16)
    plt.tight_layout()
    plt.savefig('img/age.pdf')

    # ------------------------ Rank all feats by RF ------------------------
    print('Rank all feats by RF ...')
    all_res = []
    kf = KFold(n_splits=n_fold, shuffle=True, random_state=seed)
    feature_scores = []
    for train_index, val_index in kf.split(X):
        X_train, X_val = X[train_index], X[val_index]
        y_train, y_val = y[train_index], y[val_index]
        
        # RF
        m = RF(n_estimators=100, random_state=seed)
        m.fit(X_train, y_train)
        y_pred = m.predict_proba(X_val)[:,1]
        t_res = my_eval(y_val, y_pred)
        
        feature_scores.append(m.feature_importances_)
        all_res.append(t_res['auroc'])
        
    feature_scores = np.mean(np.array(feature_scores), axis=0)
    df_imp = pd.DataFrame({'col':x_cols_all, 'score':feature_scores})
    df_imp = df_imp.merge(df_missing_rate, left_on='col', right_on='col', how='left')
    df_imp = df_imp.sort_values(by='score', ascending=False)
    df_imp.to_csv('res/imp.csv', index=False)
    
    fig = plt.figure(figsize=(6,4))
    ax = fig.add_subplot(1,1,1)
    ax.plot(df_imp.score.values, color='tab:blue', linewidth=2)
    ax.set_xlabel('Rank of Features', fontsize=20)
    ax.set_ylabel('Importance', fontsize=20)
    ax.set_ylim([0,0.08])
    # ax.set_yscale('log')
    plt.tight_layout()
    plt.savefig('img/ranking.pdf')
    
    all_res = np.array(all_res)
    res_mean = np.mean(all_res)
    res_std = np.std(all_res)
    print('RF on all features, mean: {:.4f}, std {:.4f}'.format(res_mean, res_std))
    
    # ------------------------ Select top feats by cross validation ------------------------
    print('Select top feats by cross validation ...')
    n_features = list(range(1,max_n_features+1))
    all_res = []
    for topK in tqdm(n_features, desc='feature selection'):
        tmp_res = []
        x_cols = df_imp[:topK].col.values
        X = np.nan_to_num(df[x_cols].values)
        X_test = np.nan_to_num(df_test[x_cols].values)
        kf = KFold(n_splits=n_fold, shuffle=True, random_state=seed)
        for train_index, val_index in kf.split(X):
            
            X_train, X_val = X[train_index], X[val_index]
            y_train, y_val = y[train_index], y[val_index]

            m = LR()
            m.fit(X_train, y_train)
            y_pred = m.predict_proba(X_val)[:,1]
            t_res = my_eval(y_val, y_pred)
            tmp_res.append(t_res['auroc'])

        all_res.append(tmp_res)
    
    all_res = np.array(all_res)
    res_mean = np.mean(all_res, axis=1)
    res_std = np.std(all_res, axis=1)
    res_df = pd.DataFrame(np.stack([np.array(n_features).T, res_mean, res_std], axis=1))
    res_df.columns = ['topK', 'AUROC_mean', 'AUROC_std']
    # print(res_df)
    
    fig = plt.figure(figsize=(6,4))
    ax = fig.add_subplot(1,1,1)
    plt.plot(res_df.AUROC_mean.values)
    plt.xlabel('Number of Features')
    plt.ylabel('AUROC')    
    ax.plot(res_df.AUROC_mean.values, color='tab:blue', linewidth=2)
    ax.set_xlabel('Number of Features', fontsize=20)
    ax.set_ylabel('ROC-AUC', fontsize=20)
    max_x = np.argmax(res_df.AUROC_mean)
    max_y = res_df.AUROC_mean.values[max_x]
    ax.vlines(max_x, np.min(res_df.AUROC_mean.values), max_y, linestyle='dashed', color='tab:red', linewidth=2)
    ax.set_ylim([np.min(res_df.AUROC_mean.values), np.max(res_df.AUROC_mean.values)+0.02])
    ax.annotate('22 features give the best!', (22, 0.55), xytext=(24,0.57), fontsize=14, arrowprops={'arrowstyle':'->'})
    plt.tight_layout()
    plt.savefig('img/stepwise.pdf')
        
    # ------------------------ Evaluation ------------------------
    print('Evaluation ...')
    
    topK = np.min([max_topK, np.argmax(res_df.AUROC_mean)])
    x_cols = list(df_imp[:topK].col.values)
    print('topK features: {}'.format(topK))
    print(x_cols)
    
    X = np.nan_to_num(df[x_cols].values)
    X_test = np.nan_to_num(df_test[x_cols].values)
    X_all = np.nan_to_num(df[x_cols_all].values)
    X_test_all = np.nan_to_num(df_test[x_cols_all].values)
    
    # LR
    m = LR(solver='liblinear', max_iter=10000)
    m.fit(X, y)
    print_lr_model(m, x_cols)
    y_pred_lr = m.predict_proba(X_test)[:,1]
    final_res_lr = my_eval(y_test, y_pred_lr)
    
    # RF
    m = RF(n_estimators=100, random_state=seed)
    m.fit(X, y)
    y_pred_rf = m.predict_proba(X_test)[:,1]
    final_res_rf = my_eval(y_test, y_pred_rf)

    # LR all
    m = LR(solver='liblinear', max_iter=10000)
    m.fit(X_all, y)    
    y_pred_lr_all = m.predict_proba(X_test_all)[:,1]
    final_res_lr_all = my_eval(y_test, y_pred_lr_all)
    
    # RF all
    m = RF(n_estimators=100, random_state=seed)
    m.fit(X_all, y)
    y_pred_rf_all = m.predict_proba(X_test_all)[:,1]
    final_res_rf_all = my_eval(y_test, y_pred_rf_all)
    
    # baseline prism_iii
    y_pred_prism_iii = prism_iii(df_test)
    final_res_prism_iii = my_eval(y_test, y_pred_prism_iii)
    
    # plot
    plt.figure(figsize=(6,4))
    lw = 2
    plt.plot(final_res_lr['fpr'], final_res_lr['tpr'], color='tab:red', lw=lw, label='LR selected (ROC-AUC = %0.4f)' % final_res_lr['auroc'])
    plt.plot(final_res_rf['fpr'], final_res_rf['tpr'], color='tab:blue', lw=lw, label='RF selected (ROC-AUC = %0.4f)' % final_res_rf['auroc'])
    plt.plot(final_res_lr_all['fpr'], final_res_lr_all['tpr'], color='tab:olive', lw=lw, label='LR all (ROC-AUC = %0.4f)' % final_res_lr_all['auroc'])
    plt.plot(final_res_rf_all['fpr'], final_res_rf_all['tpr'], color='tab:cyan', lw=lw, label='RF all (ROC-AUC = %0.4f)' % final_res_rf_all['auroc'])
    plt.plot(final_res_prism_iii['fpr'], final_res_prism_iii['tpr'], color='k', lw=lw, label='PRISM III (ROC-AUC = %0.4f)' % final_res_prism_iii['auroc'])
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=20)
    plt.ylabel('True Positive Rate', fontsize=20)
#     plt.title('Receiver operating characteristic', fontsize=20)
    plt.legend(loc="lower right", fontsize=10)
    plt.tight_layout()
    plt.savefig('img/roc.pdf')
        

    
    
    
    