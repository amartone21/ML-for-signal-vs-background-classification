import matplotlib
#matplotlib.use('GTKAgg')
from sklearn import svm, datasets
import warnings 
import ROOT
import root_pandas
import numpy as np
import pandas as pd
#import seaborn as sns
from pylab import rcParams
import matplotlib.pyplot as plt
from sklearn import preprocessing 
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score, accuracy_score, f1_score, confusion_matrix, auc, roc_curve
from config import fetch_configuration
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from funzioni import calibrated, uncalibrated


#import dataset
data = root_pandas.read_root('./candidati1200_3.root', 'toFormat')

#import variable dicitonary and define the ones to work with
config_dict= fetch_configuration()
config2= 'btag_pt'
config= 'all'
n_rows = data.shape[0]
truth1= 'Higgs_truth1'


''' Here I have to manage the dataset dimensions.
To optimize the performance, the ratio between signal and background is kept constant (1 to 56) 
but the dataset is reduce to  14782 
'''

signal_df_new = data.query("Higgs_truth1==1").sample(n=4*263)
fondo_df_new = (data.query("Higgs_truth1 == 0")).sample(n= 56*len(data.query("Higgs_truth1==1")  ))
df= [fondo_df_new, signal_df_new]
df_new = pd.concat(df)
df_new = df_new.query(config_dict[config]['bin'])

#splitting 
X_train, X_test = train_test_split(df_new, test_size=0.3)

#take all variables
X_train_all_variables, X_test_all_variables = X_train.query(config_dict[config]['presel']), X_test.query(config_dict[config]['presel'])

#extract only the relevant variables
X_train, X_test = X_train_all_variables[config_dict[config]['variables']], X_test_all_variables[config_dict[config]['variables']]

#create truth lables
y_train, y_test = X_train_all_variables[truth1].values, X_test_all_variables[truth1].values

#Implement LDA to reduce dimensionality
lda = LinearDiscriminantAnalysis()
trainX= lda.fit_transform(X_train,y_train)
testX= lda.transform(X_test)

#define the classifier
svc = svm.SVC(kernel='linear',probability = True)

#fitting
svc.fit(trainX,y_train)

#make predictions
y_pred = svc.predict_proba(testX)[:,1]

#retrive info for roc curve
fpr, tpr, thresholds = roc_curve(y_test, y_pred)
roc_auc = auc(fpr, tpr)
fig_roc, ax_roc = plt.subplots(figsize=(7, 5))
ax_roc.plot(fpr, tpr, lw=2, color='cyan', label='auc = %.3f' % (roc_auc))
ax_roc.plot([0, 1], [0, 1], linestyle='--', lw=2, color='k', label='random chance')
ax_roc.set_xlim([0, 1.0])
ax_roc.set_ylim([0, 1.0])
ax_roc.set_xlabel('false positive rate')
ax_roc.set_ylabel('true positive rate')
ax_roc.set_title('receiver operating curve'+ config  )
ax_roc.legend(loc="lower right")
fig_roc.savefig(  'roc'+ config  + '.png', format='png')

# evaluate efficiencies at given working points
tpr10 =[]
tpr1 =[]
t10=[]
t1=[]
fpr, tpr, thresholds = roc_curve(y_test, y_pred)
for idx in range(len(tpr)):
   if (fpr[idx])<0.1:        
      tpr10.append(tpr[idx])
      t10.append(thresholds[idx])
   if (fpr[idx])<0.01:        
      tpr1.append(tpr[idx])
      t1.append(thresholds[idx])

print(' il max al 10 in configurazione', np.amax(tpr10))
print(' il max al 1:', np.amax(tpr1))



   





