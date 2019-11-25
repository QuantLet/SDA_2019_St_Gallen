# Predicting the probability of corporate default using financial data and sentiment from annual reports
Predict the probability of default crossing sentiment analysis with accounting data on publicly listed non-financial companies in the U.S.A between 2002 and 2018 using machine learning algorithms. 

The project involved the following steps:
- Preprocessing the data
- Performing textual analysis
- Describing the data
- Identifying outliers and the 44 main features, applying oversampling techniques
- Running the machine learning models

## Data preporcessing and describing



## Performing textual analysis


## Identifying outliers and 44 main features, applying oversampling techniques
### Feature selection
The extracted dataset has a lot features (440 after initial cleaning). High dimensional datasets like this are very computational expensive to process with machine learning algorithms while a lot of the features do add very little new insights. 
To do this the following process was employed:
- An Extra Tree Forest is constructed from the original training sample 		(each decision trees learn how to best split the dataset into smaller and smaller subsets to predict the target value)
- For each decision tree, first the node importance for each node is calculated 	(gini importance). Afterwards the feature importance is calculated by dividing 	the sum of nodes importances of a feature by the overall nodes importances
- By averaging over all trees in the forest an feature importance ranking is 		created.

```
def reduce_components(X, X_name, y, top_n, do_plot):
    forest = ExtraTreesClassifier(n_estimators=250, random_state=0)
    forest.fit(X.loc[:, list(set(X.columns)-set(mergeOnColumns))], y)
    importances = forest.feature_importances_
    indices = np.argsort(importances)[::-1]
    topcols = [list(set(X.columns)-set(mergeOnColumns))[i] for i in list(indices[:top_n])]
    topimportances = [list(importances)[i] for i in list(indices[:top_n])]
    
    if do_plot:
        plt.figure(figsize=(21,7))
        plt.bar(topcols, topimportances)
        plt.xticks(rotation=90)
        plt.ylabel("Importance of feature")
        plt.title("Top {} features".format(top_n))
        plt.savefig("{}_top{}_features.png".format(X_name, top_n), transparent = True)
        plt.show()
    
    X_top = X.loc[:, ["cik"] + ["fyear"] + topcols]
    #X_top =  X.iloc[:,[list(X_all.columns).index("fyear")] + [list(X_all.columns).index("cik")] + [list(X_all.columns).index("conm")] + list(indices[:top_n])]
    return(X_top, topcols, topimportances)
 
nFeatures = 44
X_all, topcols, topimportances = reduce_components(X_all, "X_all", y, nFeatures, True)
```

### Outlier detection
Since there are a lot of outliers in the data, an Isolation Forest algorithm for implemented to identify anomalies. The percentage of outliers to be detected was  set to 10%. 
```
from sklearn.ensemble import IsolationForest
to_model_columns=x_all.columns[3:46]
 
clf=IsolationForest(n_estimators=100, max_samples='auto', contamination=float(.1), \
                        max_features = 43, bootstrap=False, n_jobs=-1, random_state=42, verbose=0)
clf.fit(x_all[to_model_columns])
pred = clf.predict(x_all[to_model_columns])
x_all['anomaly']=pred
outliers=x_all.loc[x_all['anomaly']==-1]
```


### Rebalancing the dataset
As expected the dataset is really imbalanced. By Oversampling additional data points for the minority class are generated and thus a balanced dataset is generated.
- Random Oversampling:
	over-samples the minority class by picking samples at random with 			replacement
- Synthetic Minority Over-sampling:
	over-samples the minority class by generating new artificial samples that 		are between two of the samples of the target class.
```
for i in range(len(dfs)):
 
    #Random naive over-sampling
    from imblearn.over_sampling import RandomOverSampler
 
    X_resampled, y_resampled = RandomOverSampler(random_state=0).fit_resample(dfs[i], y_train)
    X_resampled = pd.DataFrame(X_resampled, columns = dfs[i].columns)
    y_resampled = pd.DataFrame(y_resampled, columns=["bnkrpt"])
        
    X_resampled[X_all_train.columns].to_csv("{}_RNOS.csv".format(dfs_names[i]))
    y_resampled.to_csv("y{}_RNOS.csv".format(dfs_names[i][1:]))
    
    #Synthetic minority oversampling
    from imblearn.over_sampling import SMOTE
    
    X_resampled, y_resampled = SMOTE().fit_resample(dfs[i], y_train)
    X_resampled = pd.DataFrame(X_resampled, columns = dfs[i].columns)
    y_resampled = pd.DataFrame(y_resampled, columns=["bnkrpt"])
        
    X_resampled[X_all_train.columns].to_csv("{}_SMOTEOS.csv".format(dfs_names[i]))
    y_resampled.to_csv("y{}_SMOTEOS.csv".format(dfs_names[i][1:]))
```



## Running the machine learning models
