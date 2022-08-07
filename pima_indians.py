#!/usr/bin/env python
# coding: utf-8


# `Fields description:`
# - preg = Number of times pregnant
# - plas = Plasma glucose concentration a 2 hours in an oral glucose tolerance test
# - pres = Diastolic blood pressure (mm Hg)
# - skin = Triceps skin fold thickness (mm)
# - test = 2-Hour serum insulin (mu U/ml)
# - mass = Body mass index (weight in kg/(height in m)^2)
# - pedi = Diabetes pedigree function
# - age = Age (years)
# - class = Class variable (1:tested positive for diabetes, 0: tested negative for diabetes)

# In[1]:
import numpy as np
import pandas as pd
import time
import matplotlib.pyplot as plt
# get_ipython().run_line_magic('matplotlib', 'inline
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_score, GridSearchCV, RandomizedSearchCV, train_test_split, cross_val_score, ShuffleSplit
from imblearn.over_sampling import SMOTE
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, VotingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report, accuracy_score
from sklearn.impute import SimpleImputer
import joblib
import dill
# In[2]:
columns = ['preg', 'plas', 'pres', 'skin', 'test', 'mass', 'pedi', 'age', 'class']
df_bruno = pd.read_csv('pima_indians_diabetes.csv', names = columns)
# In[3]:
df_bruno.head()
# In[4]:
df_bruno.tail()
# In[5]:
df_bruno.dtypes
# In[6]:
df_bruno.info()
# In[7]:
df_bruno.describe()
# In[8]:
df_bruno.isnull().sum()
# In[9]:
sns.set(rc={"figure.figsize": (28,15)})
sns.heatmap(df_bruno.isnull(), yticklabels=False, cbar=False, cmap='viridis')
plt.show()
# #### Features with with unaccceptable 0 (zero) values
# In[10]:
# Identify columns with zeros
df_bruno.drop('class', axis = 1).eq(0).any()
# In[11]:
zero_unacceptable = ['plas', 'pres', 'skin', 'test', 'mass']
for col in zero_unacceptable:
    df_bruno[col].replace(0, np.nan, inplace=True)
# In[12]:
df_bruno.head()
# In[13]:
# Visualizing Null Values
plt.figure(figsize=(9,5))
ax = sns.barplot(x=df_bruno.isna().sum(),
                y=df_bruno.columns, orient='h')
for p in ax.patches:
    ax.annotate(text=f"{p.get_width():.0f}",
               xy=(p.get_width(), p.get_y()+p.get_height()/2),
               xytext=(5, 0), textcoords='offset points',
               ha="left", va="center")
plt.grid(False)
plt.show()
# In[14]:
sns.set(rc={"figure.figsize": (28,15)})
sns.heatmap(df_bruno.isnull(), yticklabels=False, cbar=False, cmap='viridis')
plt.show()
# #### Impute median in null-value cells
# In[15]:
imputer = SimpleImputer(strategy='mean')
# In[16]:
df_bruno[['plas', 'pres', 'skin', 'test', 'mass']] = imputer.fit_transform(df_bruno[['plas', 'pres', 'skin', 'test', 'mass']])
# In[17]:
df_bruno.describe()
# In[18]:
df_bruno['class'].value_counts()
# In[19]:
proportion = df_bruno['class'].value_counts()/len(df_bruno['class'])
# In[20]:
proportion
# In[21]:
# Check for imbalanced target variables
plt.figure(figsize = (10,6))
sns.barplot(x = [0, 1], y = proportion)
plt.xticks(np.arange(2),('False', 'True'))
plt.ylabel('Proportion')
plt.show()
# In[22]:
transformer_bruno = StandardScaler()
# In[23]:
#df_features = transformer_bruno.fit_transform(df_bruno.drop('class', axis = 1))
# In[24]:
df_features = df_bruno.drop('class', axis = 1)
# In[25]:
df_target = df_bruno['class']
# In[26]:
df_features.head()
# In[27]:
df_target
# In[28]:
X_train_bruno, X_test_bruno, y_train_bruno, y_test_bruno = train_test_split(df_features, df_target, test_size=0.3, random_state=42)
# In[29]:
X_train_bruno_tr = transformer_bruno.fit_transform(X_train_bruno)
# In[30]:
X_test_bruno_tr = transformer_bruno.transform(X_test_bruno)
# ## Oversample the minority class (1: Positive For Diabetes)
# In[31]:
def upsample_SMOTE(X_train, y_train, ratio=1.0):
    """Upsamples minority class using SMOTE.
    Ratio argument is the percentage of the upsampled minority class in relation
    to the majority class. Default is 1.0
    """
    sm = SMOTE(random_state=42, sampling_strategy=ratio)
    X_train_sm, y_train_sm = sm.fit_resample(X_train, y_train)
    print(len(X_train_sm), len(y_train_sm))
    return X_train_sm, y_train_sm
# In[32]:
# Concatenate training data back together
concat_train = pd.concat([X_train_bruno, y_train_bruno], axis = 1)
# In[33]:
concat_train.head()
# In[34]:
proportion_train = concat_train['class'].value_counts()/len(concat_train['class'])
# In[35]:
proportion_train
# In[36]:
X_train_bruno_sm, y_train_bruno_sm = upsample_SMOTE(X_train_bruno, y_train_bruno, .80) 
# In[37]:
unique, count = np.unique(y_train_bruno_sm, return_counts=True)
# In[38]:
print(np.c_[unique, count])
# In[39]:
X_train_bruno_sm_tr = transformer_bruno.fit_transform(X_train_bruno_sm)
# ## Instantiate classifiers
# In[40]:
lr_M = LogisticRegression(max_iter=1400)
rf_M = RandomForestClassifier()
svc_M = SVC()
dt_M = DecisionTreeClassifier(criterion='entropy', max_depth=42)
etc_M = ExtraTreesClassifier()
# ### Hard voting
# In[41]:
voting_hard = VotingClassifier(
                    estimators=[
                        ('logistic_reg', lr_M),
                        ('random_forest', rf_M),
                        ('svc', svc_M),
                        ('decision_trees', dt_M),
                        ('extra_trees', etc_M)
                        ],
                    voting = 'hard',
                    n_jobs = -1)
# In[42]:
classifiers = [lr_M, rf_M, svc_M, dt_M, etc_M, voting_hard]
predictions = pd.DataFrame(columns=['Logistic_Regression',
                                  'Random_Forest',
                                  'SVC',
                                  'Decision_Trees',
                                  'Extra_Trees',
                                  'Voting_Classifier'])
for count, clf in enumerate(classifiers):
#     print(count, clf)
    clf.fit(X_train_bruno_sm_tr, y_train_bruno_sm)
    predictions[predictions.columns[count]] = clf.predict(X_test_bruno_tr[:3])
# In[43]:
predictions['class'] = y_test_bruno.reset_index(drop=True)
# In[44]:
predictions
# ### Soft Voting
# In[45]:
lr_soft_M = LogisticRegression(max_iter = 1400)
rf_soft_M = RandomForestClassifier()
svc_soft_M = SVC(probability = True)
dt_soft_M = DecisionTreeClassifier(criterion='entropy', max_depth=42)
etc_soft_M = ExtraTreesClassifier()
# In[46]:
voting_soft = VotingClassifier(
                    estimators=[
                        ('logistic_reg_soft', lr_soft_M),
                        ('random_forest_soft', rf_soft_M),
                        ('svc_soft', svc_soft_M),
                        ('decision_trees_soft', dt_soft_M),
                        ('extra_trees_soft', etc_soft_M)
                        ],
                    voting = 'soft',
                    n_jobs = -1)
# In[47]:
classifiers_soft = [lr_soft_M, rf_soft_M, svc_soft_M, dt_soft_M, etc_soft_M, voting_soft]
predictions_soft = pd.DataFrame(columns=['Logistic_Regression',
                                  'Random_Forest',
                                  'SVC',
                                  'Decision_Trees',
                                  'Extra_Trees',
                                  'Voting_Classifier'])
for count, clf in enumerate(classifiers_soft):
#     print(count, clf)
    clf.fit(X_train_bruno_sm_tr, y_train_bruno_sm)
    predictions_soft[predictions_soft.columns[count]] = clf.predict(X_test_bruno_tr[:3])
predictions_soft['class'] = y_test_bruno.reset_index(drop=True)
# In[48]:
predictions_soft
# ##  Extra Trees and Decision Trees
# In[49]:
pipeline1_bruno = Pipeline([
    ('tranformer', transformer_bruno),
    ('extra_trees', etc_M)
    ])
# In[50]:
pipeline2_bruno = Pipeline([
    ('tranformer', transformer_bruno),
    ('decision_trees', dt_M)
    ])
# In[51]:
pipeline1_bruno.fit(X_train_bruno, y_train_bruno)
# In[52]:
pipeline2_bruno.fit(X_train_bruno, y_train_bruno)
# In[53]:
cv = ShuffleSplit(n_splits=10, random_state=42)
# ### Scores Pipeline 1
# #### Cross Validation with original training data
# In[54]:
scores_pipe1_1 = cross_val_score(pipeline1_bruno,
                               X_train_bruno,
                               y_train_bruno,
                               cv = cv,
                               n_jobs=-1,
                              verbose=1)
# In[55]:
scores_pipe1_1
# In[56]:
print(f'Average scores original data: {round(scores_pipe1_1.mean(),2)}')
# #### Cross Validation with upsampled training data
# In[57]:
pipeline1_bruno.fit(X_train_bruno_sm, y_train_bruno_sm)
# In[58]:
scores_pipe1_2 = cross_val_score(pipeline1_bruno,
                               X_train_bruno_sm,
                               y_train_bruno_sm,
                               cv = cv,
                               n_jobs=-1,
                              verbose=1)
# In[59]:
scores_pipe1_2
# In[60]:
print(f'Average scores upasampled data: {round(scores_pipe1_2.mean(),2)}')
# ### Scores Pipeline 2
# #### Cross Validation with original training data
# In[61]:
scores_pipe2_1 = cross_val_score(pipeline2_bruno,
                                X_train_bruno,
                                y_train_bruno,
                                cv = cv,
                                n_jobs=-1,
                                verbose=1) 
# In[62]:
print(f'Average scores original data: {round(scores_pipe2_1.mean(),2)}')
# In[63]:
pipeline2_bruno.fit(X_train_bruno_sm, y_train_bruno_sm)
# #### Cross Validation with upsampled training data
# In[64]:
scores_pipe2_2 = cross_val_score(pipeline2_bruno,
                                X_train_bruno_sm,
                                y_train_bruno_sm,
                                cv = cv,
                                n_jobs=-1,
                                verbose=1) 
# In[65]:
print(f'Average scores upsampled data: {round(scores_pipe2_2.mean(),2)}')
# #### Predictions Pipeline 1
# In[66]:
pred_pipe1 = pipeline1_bruno.predict(X_test_bruno)
# In[67]:
cm1 = confusion_matrix(y_test_bruno, pred_pipe1, labels=pipeline1_bruno.classes_)
# In[68]:
disp1 = ConfusionMatrixDisplay(confusion_matrix=cm1,
                               display_labels=pipeline1_bruno.classes_)
# In[69]:
fig, ax = plt.subplots(figsize=(6,6))
disp1.plot(ax=ax ,colorbar=False)
plt.grid(False)
plt.title('Confusion Matrix - Extra Trees')
plt.show()
# In[70]:
# Print the classification report
print('\t\tClassification Report - Extra Trees\n\n', classification_report(y_test_bruno, pred_pipe1))
# precision means what percentage of the positive predictions made were actually correct.</br> 
# `TP/(TP+FP)`
#
# Recall in simple terms means, what percentage of actual positive predictions were correctly classified by the classifier.</br> 
# `TP/(TP+FN)`
#
# F1 score can also be described as the harmonic mean or weighted average of precision and recall.</br> 
# `2x((precision x recall) / (precision + recall))`

# #### Predictions Pipeline 2
# In[71]:
pred_pipe2 = pipeline2_bruno.predict(X_test_bruno)
# In[72]:
cm2 = confusion_matrix(y_test_bruno, pred_pipe2, labels=pipeline2_bruno.classes_)
# In[73]:
disp2 = ConfusionMatrixDisplay(confusion_matrix=cm2,
                               display_labels=pipeline2_bruno.classes_)
# In[74]:
fig, ax = plt.subplots(figsize=(6,6))
disp2.plot(ax=ax ,colorbar=False, cmap='BuGn')
plt.grid(False)
plt.title('Confusion Matrix - Decision Trees')
plt.show()
# In[75]:
# Print the classification report
print('\t\tClassification Report - Decision Trees\n\n', classification_report(y_test_bruno, pred_pipe2))
# precision means what percentage of the positive predictions made were actually correct.</br> 
# `TP/(TP+FP)`
#
# Recall in simple terms means, what percentage of actual positive predictions were correctly classified by the classifier.</br> 
# `TP/(TP+FN)`
#
# F1 score can also be described as the harmonic mean or weighted average of precision and recall.</br> 
# `2x((precision x recall) / (precision + recall))`

# ### Randomized GridSearch

# #### Extra Trees
# In[76]:
pipeline1_bruno_v2 = Pipeline([
    ('tranformer', transformer_bruno),
    ('extra_trees', etc_M)
    ])
# In[77]:
joblib.dump(pipeline1_bruno_v2, "pipeline_extra_trees.pkl")
# In[78]:
parameters={'extra_trees__n_estimators' : range(10,3000,20),
            'extra_trees__max_depth': range(1,1000,2)}
# In[79]:
rand_gridsearch_bruno = RandomizedSearchCV(estimator = pipeline1_bruno_v2,
                                       param_distributions = parameters,
                                       scoring = 'accuracy',
                                       cv = 5,
                                       n_iter = 50,
                                       refit = True,
                                       n_jobs = -1,
                                       verbose = 3,
                                       random_state = 42)
# In[80]:
rand_gridsearch_bruno.fit(X_train_bruno_sm, y_train_bruno_sm)
# In[81]:
# Best hyperparameters
print("tuned hpyerparameters :(best parameters) \n", rand_gridsearch_bruno.best_params_)
# In[82]:
# Store the best model
best_model = rand_gridsearch_bruno.best_estimator_
# In[83]:
joblib.dump(best_model, 'best_model_extra_trees.pkl')
# In[84]:
# Make new predictions with the tuned model
final_pred = best_model.predict(X_test_bruno)
# In[85]:
best_model.score(X_test_bruno, y_test_bruno)
# In[86]:
cm3 = confusion_matrix(y_test_bruno, final_pred, labels=pipeline1_bruno.classes_)
# In[87]:
disp3 = ConfusionMatrixDisplay(confusion_matrix=cm3,
                               display_labels=pipeline1_bruno.classes_)
# In[88]:
fig, ax = plt.subplots(figsize=(6,6))
disp3.plot(ax=ax ,colorbar=False, cmap='BuGn')
plt.grid(False)
plt.title('Confusion Matrix - Tuned Extra Trees')
plt.show()
# In[89]:
# Print the classification report
print('\t\tClassification Report - Tuned Extra Trees\n\n', classification_report(y_test_bruno, final_pred))
# ### Experimenting with SVC
# In[90]:
svc = SVC()
# In[91]:
pipeline1_svc = Pipeline([
    ('tranformer', transformer_bruno),
    ('svc', svc)
    ])
# In[92]:
# Dictionary with parameters names (str) as keys and lists of parameter settings to try as values.
param_grid = {'svc__kernel': ['linear', 'rbf', 'poly'],
              'svc__C': [0.01, 0.1, 1, 10, 100],
              'svc__gamma': [0.01, 0.03, 0.1, 0.3, 1.0, 3.0],
              'svc__degree': [2, 3]}
# In[93]:
rand_gridsearch_svc = RandomizedSearchCV(estimator = pipeline1_svc,
                                       param_distributions = param_grid,
                                       scoring = 'accuracy',
                                       cv = 5,
                                       n_iter = 50,
                                       refit = True,
                                       n_jobs = -1,
                                       verbose = 3,
                                       random_state = 42)
# In[94]:
rand_gridsearch_svc.fit(X_train_bruno_sm, y_train_bruno_sm)
# In[95]:
# Best hyperparameters
print("tuned hpyerparameters :(best parameters) \n", rand_gridsearch_svc.best_params_)
# In[96]:
# Store the best model
best_model_svc = rand_gridsearch_svc.best_estimator_
# In[97]:
# Make new predictions with the tuned model
svc_pred = best_model_svc.predict(X_test_bruno)
# In[98]:
best_model_svc.score(X_test_bruno, y_test_bruno)
# In[99]:
# Print the classification report
print('\t\tClassification Report - Tuned SVC\n\n', classification_report(y_test_bruno, svc_pred))
# ### Experimenting with Logistic Regression
# In[100]:
log = LogisticRegression()
# In[101]:
pipeline1_log = Pipeline([
    ('tranformer', transformer_bruno),
    ('log', log)
    ])
# In[102]:
# Dictionary with parameters names (str) as keys and lists of parameter settings to try as values.
param_grid_log = {'log__penalty': ['l1', 'l2', 'elasticnet'],
                  'log__C': [0.01, 0.1, 1, 10, 100],
                  'log__solver': ['newton-cg', 'lbfgs', 'liblinear', 'saga']}
# In[103]:
rand_gridsearch_log = RandomizedSearchCV(estimator = pipeline1_log,
                                       param_distributions = param_grid_log,
                                       scoring = 'accuracy',
                                       cv = 5,
                                       n_iter = 50,
                                       refit = True,
                                       n_jobs = -1,
                                       verbose = 3,
                                       random_state = 42)
# In[104]:
rand_gridsearch_log.fit(X_train_bruno_sm, y_train_bruno_sm)
# In[105]:
# Best hyperparameters
print("tuned hpyerparameters :(best parameters) \n", rand_gridsearch_log.best_params_)
# In[106]:
# Store the best model
best_model_log = rand_gridsearch_log.best_estimator_
# In[107]:
# Make new predictions with the tuned model
log_pred = best_model_log.predict(X_test_bruno)
# In[108]:
best_model_log.score(X_test_bruno, y_test_bruno)
# In[109]:
# Print the classification report
print('\t\tClassification Report - Tuned Logistic Regression\n\n', classification_report(y_test_bruno, log_pred))
# ### Experimenting with Random Forest
# In[118]:
rf2 = RandomForestClassifier()
# In[119]:
pipeline1_rf2 = Pipeline([
    ('tranformer', transformer_bruno),
    ('rf2', rf2)
    ])
# In[120]:
# Dictionary with parameters names (str) as keys and lists of parameter settings to try as values.
param_grid_rf2 = {'rf2__n_estimators': [100, 150, 200, 300, 400, 500],
                  'rf2__criterion': ['gini', 'entropy', 'log_loss'],
                  'rf2__max_features': ['sqrt', 'log2'],
                  'rf2__max_depth': [int(x) for x in np.linspace(10, 110, num=11)],
                  'rf2__ccp_alpha': [x for x in np.arange(0.0, 0.035, 0.005)]}
# In[121]:
rand_gridsearch_rf2 = RandomizedSearchCV(estimator = pipeline1_rf2,
                                       param_distributions = param_grid_rf2,
                                       scoring = 'accuracy',
                                       cv = 5,
                                       n_iter = 50,
                                       refit = True,
                                       n_jobs = -1,
                                       verbose = 3,
                                       random_state = 42)
# In[122]:
rand_gridsearch_rf2.fit(X_train_bruno_sm, y_train_bruno_sm)
# In[123]:
# Best hyperparameters
print("tuned hpyerparameters :(best parameters) \n", rand_gridsearch_rf2.best_params_)
# In[124]:
# Store the best model
best_model_rf2 = rand_gridsearch_rf2.best_estimator_
# In[128]:
joblib.dump(best_model_rf2, 'best_model_random_forest.pkl')
# In[125]:
# Make new predictions with the tuned model
rf2_pred = best_model_rf2.predict(X_test_bruno)
# In[126]:
best_model_rf2.score(X_test_bruno, y_test_bruno)
# In[127]:
# Print the classification report
print('\t\tClassification Report - Tuned Logistic Regression\n\n', classification_report(y_test_bruno, rf2_pred))
