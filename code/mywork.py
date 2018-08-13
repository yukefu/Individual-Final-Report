import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
import tensorflow as ts

#from pandas.core.frame import DataFrame
train = pd.read_csv("/Users/yukifu/Desktop/celine/GWU/Classes/dats6202/final project/train.csv")

#------------------------------------------------- label encode part----------------------------------------------
# show how many categories the SexuponOutcome have
pd.Categorical(train.SexuponOutcome)
#5 categories: Intact Female, Intact Male, Neutered Male, Spayed Female, Unknown

# male:0, female:1, unknown:2
sex_mapping={"Intact Male":"0","Neutered Male":"0","Intact Female":"1","Spayed Female":"1","Unknown":"2"}
train["sex"]="1" #create a new feature, sex
train.sex=train.SexuponOutcome.map(sex_mapping)

# intact:0, spayed:1, unknown:2
fertility_mapping={"Intact Male":"0","Intact Female":"0","Neutered Male":"1","Spayed Female":"1","Unknown":"2"}
train["fertility"]="1" #create a new feature, fertility
train.fertility=train.SexuponOutcome.map(fertility_mapping)

# show how many categories the Color have
color=pd.Categorical(train.Color) # there are 366 categories in Color
train["MixColor"]=pd.Series(train.Color.str.contains('/')).astype('int') #create a new feature, MixColor: if mix, then 1, otherwise, 0

#create a new feature, color, only remain first color
train["color"]=train.Color.str.split("/")
train.color=[train.color[i][0] for i in range(len(train.color))]


ColorCounts=train.color.value_counts()
color_mapping={"Black": "0" , "White":"1", "Brown Tabby": "2" , "Brown": "3" , "Tan": "4" , "Orange Tabby": "5" ,
               "Blue": "6" ,"Tricolor": "7" ,"Red": "8" , "Brown Brindle": "9" , "Blue Tabby": "10" ,
               "Tortie":"11", "Calico": "12","Chocolate": "13" , "Torbie":"14", "Sable": "15" ,
               "Cream Tabby": "16", "Buff":"17", "Yellow": "18" , "Gray": "19" ,"Cream":"20",
               "Fawn": "21", "Lynx Point": "22" , "Blue Merle": "23" , "Seal Point": "24" ,"Black Brindle": "25" ,
               "Flame Point":"26" , "Gold": "27", "Brown Merle": "28" , "Black Smoke": "29" ,"Black Tabby":"30" ,
               "Silver": "31" , "Red Merle": "32" , "Gray Tabby": "33", "Blue Tick": "34" ,"Orange": "35",
               "Silver Tabby": "36" , "Red Tick": "37", "Lilac Point": "38" , "Tortie Point": "39","Yellow Brindle": "40" ,
               "Blue Point": "41" , "Calico Point": "42" , "Apricot": "43","Chocolate Point": "44" , "Blue Cream": "45" ,
               "Liver": "46" , "Blue Tiger": "47" , "Blue Smoke": "48","Liver Tick": "49", "Brown Tiger": "50" ,
               "Black Tiger": "51" , "Agouti": "52" , "Silver Lynx Point": "53", "Orange Tiger": "54", "Ruddy": "55" , "Pink": "56"}
#create a new feature: colorC, what is the encoding color
train["colorC"]="1"
train.colorC=train.color.map(color_mapping)

train.AgeuponOutcome.value_counts()
train["agenumber"]=train.AgeuponOutcome.str.split(expand=True)[0].astype("float")
train["ageperiod"]=train.AgeuponOutcome.str.split(expand=True)[1]

#create a new feature, age, age by day
age=[]
for i in range(len(train.AnimalID)):
    if train.ageperiod[i]=="day" or train.ageperiod[i]=="days":
        age.append(1*train.agenumber[i])
    elif train.ageperiod[i]=="week" or train.ageperiod[i]=="weeks":
        age.append(7*train.agenumber[i])
    elif train.ageperiod[i]=="month" or train.ageperiod[i]=="months":
        age.append(30*train.agenumber[i])
    elif train.ageperiod[i]=="year" or train.ageperiod[i]=="years":
        age.append(365*train.agenumber[i])
    else: age.append("NaN")
train["age"]=age

ageC=[]#create a new feature, ageC, that is the age classifier
for i in range(len(train.AnimalID)):
    if train.ageperiod[i]=="year" or train.ageperiod[i]=="years":
        ageC.append(train.agenumber[i].astype("int"))
    elif train.age[i]=="NaN":
        ageC.append("NaN")
    else: ageC.append(0)
train["ageC"]=ageC



train["MainBreed"]=train.Breed.str.split("/") #create a new variable, MainBreed
train.MainBreed=[train.MainBreed[i][0] for i in range(len(train.MainBreed))]

#create a new variable, MixBreed
train["MixBreed"]=pd.Series(train.Breed.str.contains('Mix')+train.Breed.str.contains('/')).astype("int")
class_le = LabelEncoder()
train.MainBreed = class_le.fit_transform(train.MainBreed)#fit and transform the class

# dog:1 cat:2
animal_mapping={"Dog":"1","Cat":"2"}
#create a new feature, animal
train["animal"]="1"
train.animal=train.AnimalType.map(animal_mapping)

#create a new feature, HaveName
train["HaveName"]=abs(pd.Series(train.Name.isnull()).astype("int")-1)


#20180810
train.OutcomeType.value_counts()
outcometype_mapping={"Adoption":"0","Transfer":"1","Return_to_owner":"2","Euthanasia":"3","Died":"4"}
#create a new target: outcome
train["outcome"]="1"
train.outcome=train.OutcomeType.map(outcometype_mapping)

#--------------------------------------missing data part-----------------------------------------
################missing data--remove
list=train[(train.age=="NaN")].index.tolist()
train=train.drop(list) #remove 18 rows

train.sex=train.sex.fillna('nan')
list=train[(train.sex=='nan')].index.tolist()
train=train.drop(list) #remove 1 row
train.to_csv("c")


#------------------------------------one hot encoding & merge data------------------------------------
train = pd.read_csv("/Users/yukifu/Desktop/celine/GWU/Classes/dats6202/final project/trainK.csv")
train.columns
train_copy = train.copy(deep=True)
train_copy=train_copy.drop(columns=[ 'sex', 'fertility',
       'MixColor', 'colorC',  'ageC',
       'MixBreed', 'animal', 'HaveName','MainBreedMix','Main Breed'])

train=train.drop(columns=['Aggressive', 'At Vet', 'Barn', 'Behavior',
       'Court/Investigation', 'Enroute', 'Foster', 'In Foster', 'In Kennel',
       'In Surgery', 'Medical', 'Offsite', 'Partner', 'Rabies Risk', 'SCRP',
       'Suffering','BreedName','color','agenumber', 'ageperiod', 'age','outcome','Target','MainBreedMix'])

train.iloc[:,1:10]=train.iloc[:,1:10].astype("int")
train.iloc[:,1:10]=train.iloc[:,1:10].astype("object")
train_onehot=pd.get_dummies(train.iloc[:,1:10])
train_onehot["AnimalID"]=train["AnimalID"]
train=pd.merge(train_onehot, train_copy, on='AnimalID',how='outer')
train.outcome=train.outcome.astype("int")
train.outcome=train.outcome.astype("object")
train.Target=train.Target.astype("int")
train.Target=train.Target.astype("object")
train.to_csv("/Users/yukifu/Desktop/celine/GWU/Classes/dats6202/final project/train_onehot.csv")



#--------------------------------------------------------------------------Target part--------------------------------------------------------------------------
train=pd.read_csv("/Users/yukifu/Desktop/celine/GWU/Classes/dats6202/final project/train_onehot_target.csv")
train=train.drop(columns=['AnimalID','Aggressive', 'At Vet', 'Barn', 'Behavior',
       'Court/Investigation', 'Enroute', 'Foster', 'In Foster', 'In Kennel',
       'In Surgery', 'Medical', 'Offsite', 'Partner', 'Rabies Risk', 'SCRP',
       'Suffering','BreedName','color','agenumber', 'ageperiod', 'age','outcome'])

from sklearn.model_selection import train_test_split
X=train.values[:,0:312]
Y=train['Target'].values
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=0)


from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt


#---------------------------Grid Search---------------------------------------------------
from sklearn.model_selection import GridSearchCV
tree_para = {'criterion':['gini','entropy'],'max_depth':[1,2,3,4,5,6,7,8,9,10,11,12,15,20,30,40,50,70,80,90,100,110,120,130,140,150],
             'min_samples_leaf':[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,19,20]}
clf = GridSearchCV(DecisionTreeClassifier(), tree_para, cv=5)
clf.fit(X, Y)
clf.best_params_
# the best criterion is entropy, max_depth=15, min_samples_leaf=11


#-----------------------------------------Entropy Decision Tree-------------------------------
# perform training with entropy.
# Decision tree with entropy
clf_entropy = DecisionTreeClassifier(criterion="entropy", random_state=100, max_depth=15, min_samples_leaf=11)
# Performing training
clf_entropy.fit(X_train, Y_train)
#%%-----------------------------------------------------------------------
# make predictions
# predicton on test using entropy
Y_pred_entropy = clf_entropy.predict(X_test)
#%%-----------------------------------------------------------------------
# calculate metrics entropy model
print("\n")
print("Results Using Entropy: \n")
print("Classification Report: ")
print(classification_report(Y_test,Y_pred_entropy))
print("\n")
print("Accuracy : ", accuracy_score(Y_test, Y_pred_entropy) * 100)
print ('-'*80 + '\n')
#%%-----------------------------------------------------------------------
# confusion matrix for entropy model
conf_matrix = confusion_matrix(Y_test, Y_pred_entropy)
class_names = train.target.unique()
df_cm = pd.DataFrame(conf_matrix, index=class_names, columns=class_names )

plt.figure(figsize=(5,5))
hm = sns.heatmap(df_cm, cbar=False, annot=True, square=True, fmt='d', annot_kws={'size': 20},
                 yticklabels=df_cm.columns, xticklabels=df_cm.columns)
hm.yaxis.set_ticklabels(hm.yaxis.get_ticklabels(), rotation=0, ha='right', fontsize=20)
hm.xaxis.set_ticklabels(hm.xaxis.get_ticklabels(), rotation=0, ha='right', fontsize=20)
plt.ylabel('True label',fontsize=20)
plt.xlabel('Predicted label',fontsize=20)
plt.tight_layout()
plt.show()

#%%-----------------------------------------------------------------------
# display decision tree
from pydotplus import graph_from_dot_data
from sklearn.tree import export_graphviz
import webbrowser

dot_data = export_graphviz(clf_entropy, filled=True, rounded=True, class_names=class_names,
                           feature_names=train.iloc[:, :-2].columns, out_file=None)

graph = graph_from_dot_data(dot_data)
graph.write_pdf("decision_tree_entropy.pdf")
webbrowser.open_new(r'decision_tree_entropy.pdf')

print ('-'*40 + 'End Console' + '-'*40 + '\n')
#%%-----------------------------------------------------------------------
# display important features
importances=clf_entropy.feature_importances_
for k,v in sorted(zip(map(lambda x: round(x, 5), importances), train.columns), reverse=True):
    print (v + ": " + str(k))

from sklearn.metrics import roc_curve,auc

predictions = clf_entropy.predict_proba(X_test)
false_positive_rate, true_positive_rate, _ = roc_curve(Y_test, predictions[:,1])
roc_auc = auc(false_positive_rate, true_positive_rate)

plt.title('Receiver Operating Characteristic')
plt.plot(false_positive_rate, true_positive_rate, 'b',
label='AUC = %0.2f'% roc_auc)
plt.legend(loc='lower right')
plt.plot([0,1],[0,1],'r--')
plt.xlim([-0.1,1.2])
plt.ylim([-0.1,1.2])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()

#--------------------------------------------------outcome part--------------------------------------------------------------------------

train=pd.read_csv("/Users/yukifu/Desktop/celine/GWU/Classes/dats6202/final project/train_onehot_outcome.csv")
train=train.drop(columns=['AnimalID','Aggressive', 'At Vet', 'Barn', 'Behavior',
       'Court/Investigation', 'Enroute', 'Foster', 'In Foster', 'In Kennel',
       'In Surgery', 'Medical', 'Offsite', 'Partner', 'Rabies Risk', 'SCRP',
       'Suffering','BreedName','color','agenumber', 'ageperiod', 'age','Target'])

from sklearn.model_selection import train_test_split
X=train.values[:,0:312]
Y=train['outcome'].values
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=0)


#------------------------------------Grid Search---------------------------------------------------
from sklearn.model_selection import GridSearchCV
tree_para = {'criterion':['gini','entropy'],'max_depth':[1,2,3,4,5,6,7,8,9,10,11,12,15,20,30,40,50,70,80,90,100,110,120,130,140,150],
             'min_samples_leaf':[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,19,20]}
clf = GridSearchCV(DecisionTreeClassifier(), tree_para, cv=5)
clf.fit(X, Y)
clf.best_params_
# the best criterion is entropy, max_depth=7, min_samples_leaf=19


#----------------------------Entropy Decision Tree--------------------------------------------------
# perform training with entropy.
# Decision tree with entropy
clf_entropy = DecisionTreeClassifier(criterion="entropy", random_state=100, max_depth=7, min_samples_leaf=19)
# Performing training
clf_entropy.fit(X_train, Y_train)
#%%-----------------------------------------------------------------------
# make predictions
# predicton on test using entropy
Y_pred_entropy = clf_entropy.predict(X_test)
#%%-----------------------------------------------------------------------
# calculate metrics entropy model
print("\n")
print("Results Using Entropy: \n")
print("Classification Report: ")
print(classification_report(Y_test,Y_pred_entropy))
print("\n")
print("Accuracy : ", accuracy_score(Y_test, Y_pred_entropy) * 100)
print ('-'*80 + '\n')
#%%-----------------------------------------------------------------------
# confusion matrix for entropy model
conf_matrix = confusion_matrix(Y_test, Y_pred_entropy)
class_names = train.outcometype.unique()
df_cm = pd.DataFrame(conf_matrix, index=class_names, columns=class_names )

plt.figure(figsize=(5,5))
hm = sns.heatmap(df_cm, cbar=False, annot=True, square=True, fmt='d', annot_kws={'size': 10},
                 yticklabels=df_cm.columns, xticklabels=df_cm.columns)
hm.yaxis.set_ticklabels(hm.yaxis.get_ticklabels(), rotation=0, ha='right', fontsize=8)
hm.xaxis.set_ticklabels(hm.xaxis.get_ticklabels(), rotation=0, ha='right', fontsize=8)
plt.ylabel('True label',fontsize=10)
plt.xlabel('Predicted label',fontsize=10)
plt.tight_layout()
plt.show()

#%%-----------------------------------------------------------------------
# display decision tree
dot_data = export_graphviz(clf_entropy, filled=True, rounded=True, class_names=class_names,
                           feature_names=train.iloc[:, :-2].columns, out_file=None)

graph = graph_from_dot_data(dot_data)
graph.write_pdf("decision_tree_entropy_outcome.pdf")
webbrowser.open_new(r'decision_tree_entropy_outcome.pdf')

print ('-'*40 + 'End Console' + '-'*40 + '\n')
#%%-----------------------------------------------------------------------
# display important features
importances=clf_entropy.feature_importances_
for k,v in sorted(zip(map(lambda x: round(x, 5), importances), train.columns), reverse=True):
    print (v + ": " + str(k))




#---------------------------------------------------bar chart part-------------------------------------
train=pd.read_csv("/Users/yukifu/Desktop/celine/GWU/Classes/dats6202/final project/train_20180810.csv")

train["age1C"]='1'
for i in range(len(train.OutcomeType)):
    if train.ageC[i]==0:
        train["age1C"][i]="0"

import seaborn
ct1 = pd.crosstab(train.OutcomeType, train.age1C)
plt.style.use('seaborn-muted')
plt.rcParams['figure.figsize'] = (10, 6)
plt.rc('legend',**{'fontsize':18})
ct1.plot.bar(stacked=True)
ax = plt.gca()
ax.tick_params(axis = 'x', which = 'major', labelsize = 10)
leg = ax.legend()
leg.set_title('agetype',prop={'size':15})
plt.show()
