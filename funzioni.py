import ROOT
import numpy as np
import pandas as pd
import seaborn as sns
pd.set_option('display.max_columns', 500)
import matplotlib.pyplot as plt
plt.rcParams.update({'figure.max_open_warning': 0})
plt.rcParams['figure.figsize'] = (14, 7)
import root_pandas
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score, accuracy_score, f1_score, confusion_matrix, auc, roc_curve
from sklearn.preprocessing import StandardScaler
from config import fetch_configuration
from matplotlib.colors import ListedColormap
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import learning_curve
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_val_score

# La funzione crea il modello e restituisce CV score, train score e i dati della LC
def learn_curve(X,y,c):
    
    # Scaling delle features
    sc = StandardScaler() 
     # LogisticRegression model
    log_reg = LogisticRegression(max_iter=200,random_state=11,C=c)
    # Pipeline con scaling e classificazione
    lr = Pipeline(steps=(['scaler',sc],
                        ['classifier',log_reg]))
    
    # Creo StratifiedKFold con 5 folds
    cv = StratifiedKFold(n_splits=5,random_state=11,shuffle=True) 
    # Salvo i CV scores di ogni of each fold
    cv_scores = cross_val_score(lr,X,y,scoring="accuracy",cv=cv) 
    
    # Fitting del modello
    lr.fit(X,y) 
    #Scoring del modello sul train set
    train_score = lr.score(X,y) 
    
    #Creo la learning curve
    train_size,train_scores,test_scores = learning_curve(estimator=lr,X=X,y=y,cv=cv,scoring="accuracy",random_state=11)
    
    #converto accuracy score in  misclassification rate
    train_scores = 1-np.mean(train_scores,axis=1)
    test_scores = 1-np.mean(test_scores,axis=1)
    
    lc = pd.DataFrame({"Training_size":train_size,"Training_loss":train_scores,"Validation_loss":test_scores}).melt(id_vars="Training_size")
    return {"cv_scores":cv_scores,
           "train_score":train_score,
           "learning_curve":lc}

