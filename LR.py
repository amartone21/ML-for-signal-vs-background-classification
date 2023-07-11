import ROOT
import numpy as np
import pandas as pd
pd.set_option('display.max_columns', 500)
import matplotlib.pyplot as plt
plt.rcParams.update({'figure.max_open_warning': 0})
plt.rcParams['figure.figsize'] = (14, 7)
import root_pandas
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score, accuracy_score, f1_score, confusion_matrix, auc, roc_curve
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier 
from config import fetch_configuration
from matplotlib.colors import ListedColormap
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression
from funzioni import learn_curve 
import seaborn as sns

#importo dal dizionario le variabili che mi servono
config_dict=fetch_configuration()
conf1= '100'
conf2 = 'U250'
conf3 = 'O250'

config= 'all'

#scelgo se analizzare ak4 o ak8
truth1='Higgs_truth1'  


#importo il dataset
data = root_pandas.read_root('./candidati1200_3.root', 'toFormat')

'''
#splitting 
X_train, X_test = train_test_split(data, test_size=0.3)
''' 

#Prendo X_train e X_test con tutte le colonne 
X_all  = data.query(config_dict[config]['presel'])
X = X_all[config_dict[config]['variables']]

#creo le etichette di verita
y = X_all[truth1].values




#questo e' un test
if(np.count_nonzero(y) == 0 | np.count_nonzero(y_test) == 0):
   print("No data in configuration: ", config)
   
lc = learn_curve(X,y,1)
print('Cross Validation Accuracies:', list(lc["cv_scores"]))
print ('Mean Cross Validation Accuracy:', np.mean(lc["cv_scores"]))
print ('Standard Deviation of Cross Validation Accuracy:', np.std(lc["cv_scores"]))
print ('Training Accuracy:',lc["train_score"])
sns.lineplot(data=lc["learning_curve"],x="Training_size",y="value",hue="variable")
plt.title("Learning Curve ")
plt.ylabel("Misclassification Rate/Loss");
plt.show()



#roc curve
fpr, tpr, thresholds = roc_curve(y_test, y_pred)
fpr_train, tpr_train, thresholds_train = roc_curve(y_train, y_pred)
roc_auc = auc(fpr, tpr)
fig_roc, ax_roc = plt.subplots(figsize=(7, 5))
ax_roc.plot(fpr, tpr, lw=2, color='cyan', label='auc = %.3f' % (roc_auc))
ax_roc.plot([0, 1], [0, 1], linestyle='--', lw=2, color='k', label='random chance')
ax_roc.set_xlim([0, 1.0])
ax_roc.set_ylim([0, 1.0])
ax_roc.set_xlabel('false positive rate')
ax_roc.set_ylabel('true positive rate')
ax_roc.set_title('receiver operating curve' )
ax_roc.legend(loc="lower right")
fig_roc.savefig(  'roc.png', format='png')

# evaluate predictions
tpr10 =[]
tpr1 =[]
t10=[]
t1=[]

for idx in range(len(tpr)):
   if (fpr[idx])<0.1:
      tpr10.append(tpr[idx])
      t10.append(thresholds[idx])
   if (fpr[idx])<0.01:
      tpr1.append(tpr[idx])
      t1.append(thresholds[idx])

print(' il max al 10 in configurazione', np.amax(tpr10))
print(' il max al 1:', np.amax(tpr1))
#salvo i tpr
np.savetxt('tpr10.csv', tpr10, delimiter=',')
np.savetxt('tpr1.csv', tpr1, delimiter=',')

#salvo le soglie
np.savetxt('tpr_soglie10.csv', t10, delimiter=',')
np.savetxt('tpr_soglie1.csv', t1, delimiter=',')

#salvo i fpr
np.savetxt('fpr.csv', fpr, delimiter=',')
