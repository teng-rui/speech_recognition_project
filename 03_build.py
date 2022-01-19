# Build the model.
# Hyper selection is done so the codes are comment. The retained model uses selected best parameter.
# 
# Usage: python3 ./03_build.py [--algo SVM/RF]


import argparse

from joblib import dump
import numpy as np
import sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import GridSearchCV, PredefinedSplit
from sklearn.svm import SVC


device='cuda'

    
if __name__ == "__main__":
    labels = np.array(['ARA', 'CHI', 'FRE', 'GER', 'HIN', 'ITA', 'JPN', 'KOR', 'SPA', 'TEL', 'TUR'], dtype='U3')


    # load data
    with np.load('dataset_train_clean.npz') as dataset:
        inputs_train = dataset['inputs']
        labels_train = dataset['labels']

    with np.load('dataset_devel_clean.npz') as dataset:
        inputs_devel = dataset['inputs']
        labels_devel = dataset['labels']
    
    with np.load('dataset_test_clean.npz') as dataset:
        inputs_test = dataset['inputs']
        labels_test = dataset['labels']
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--algo', default='SVM')
    args = parser.parse_args()
        
    inputs_combine=np.concatenate((inputs_train,inputs_devel))
    labels_combine=np.concatenate((labels_train,labels_devel))
    if args.algo=='SVM':
        print('training SVM')
        
        '''
        svc = SVC()
        c_range = [1,0.1,0.01,0.001]
        param_grid = {'kernel': ['rbf','linear','poly'], 'C': c_range}
        grid = GridSearchCV(svc, param_grid, cv=5,scoring='recall_macro')
        clf = grid.fit(inputs_combine, labels_combine)
        print(clf.cv_results_)
        print('best_params',clf.best_params_)
        
        svc_retrain=SVC(**clf.best_params_)
        
        '''
        
        svc_retrain=SVC(kernel='linear',C=0.01)
        svc_retrain.fit(inputs_combine, labels_combine)
        predicts=svc_retrain.predict(inputs_test)
        score=sklearn.metrics.recall_score(labels_test,predicts,average='macro')
        print('final UAR',score)
        print(confusion_matrix(labels_test,predicts,labels=labels))

        dump(svc_retrain, 'svm.joblib')
        
    if args.algo=='RF':
        print('training Randon Forest')
        
        #hyper selection
        '''
        rf = RandomForestClassifier(max_depth=2, random_state=0)
        n_estimators = [2000,3000,4000,5000]
        max_depth=[2]
        param_grid = {'n_estimators': n_estimators}#, 'max_depth': max_depth}
        
        grid = GridSearchCV(rf, param_grid, cv=5,scoring='recall_macro')
        clf = grid.fit(inputs_combine, labels_combine)
        print(clf.cv_results_)
        print(clf.best_params_)
        rf_retrain=RandomForestClassifier(**clf.best_params_)
        '''
        rf_retrain=RandomForestClassifier(n_estimators=5000)
        rf_retrain.fit(inputs_combine, labels_combine)
        predicts=rf_retrain.predict(inputs_test)
        score=sklearn.metrics.recall_score(labels_test,predicts,average='macro')
        print('final UAR',score)

        print(confusion_matrix(labels_test,predicts,labels=labels))
        
        dump(rf_retrain, 'rf.joblib')
        