import torch
def train(model,loss, train_loader,optimizer, epoch,alt_train_loaders=[],writer=None):
    model.train()
    training_loss = 0.0
    for (x, y) in train_loader:
        
        if len(alt_train_loaders):
            for AD in alt_train_loaders:
                (_x, _y)= next(iter(AD))
                x=torch.concat((x,_x),0)
                y=torch.concat((y,_y),0)
        output = model(x)
        l=loss(output, torch.nn.functional.one_hot(y.long(),2).float())
        optimizer.zero_grad()
        l.backward()
        optimizer.step()
        training_loss += l.item()
    if writer:
        writer.add_scalar("training_loss", training_loss, epoch)

from sklearn.metrics import confusion_matrix, roc_auc_score, multilabel_confusion_matrix
from pynico_eros_montin import stats as st


import numpy as np

def testPrediction(Ygt, Yhat,labels=None):
    O = []
    if labels is None:
        labels=np.unique(Ygt)
    C=multilabel_confusion_matrix(Ygt.flatten(), Yhat.flatten(),labels=labels)
    for c,l in zip(C,labels):
        tn_, fp_, fn_, tp_ = c.ravel()
        o = {"accuracy": st.accuracyFromConfusion(c),
             "specificity": st.specificityFromConfusion(c),
             "sensitivity": st.sensitivityFromConfusion(c),
             "tn": tn_,
             "tp": tp_,
             "fp": fp_,
             "fn": fn_,
             "label":l
             }
        O.append(o)

    return O


import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

def zScoreFeartures(features):
    scaler = StandardScaler()
    features = pd.DataFrame(scaler.fit_transform(features), columns=features.columns)
    return features

def filterFeaturesByMAD(features):
    """
    Calculate Median Absolute Deviation (MAD) for each feature.
    Discard features with MAD equal to zero.
    """
    mad_values = calculate_df_mad(features)
    return features.loc[:, mad_values != 0]


def calculate_df_mad(df):
    """
    Calculate the Median Absolute Deviation (MAD) for each column of the DataFrame.
    
    Parameters:
    df (pd.DataFrame): DataFrame with numerical values
    
    Returns:
    pd.Series: MAD for each column
    """
    # Calculate MAD for each column (axis=0)
    mad = df.apply(lambda x: np.median(np.abs(x - np.median(x))), axis=0)
    return mad

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelBinarizer

def labelbinarizer(y):
    lb = LabelBinarizer()
    y_bin = lb.fit_transform(y)
    return y_bin

# def filterFeaturesByAUC(X, y, feature_cols=None, threshold=0.580,returnAUC=False):
#     """
#     Calculate the AUC-ROC for each feature in a multi-class classification problem and select features with AUC >= threshold.
    
#     Parameters:
#     X (pd.DataFrame): DataFrame with feature values
#     y (pd.Series): Series with class labels (for a three-class problem)
#     feature_cols (list): List of column names that contain the feature values
#     threshold (float): The AUC threshold for selecting predictive features (default is 0.580)

#     Returns:
#     pd.DataFrame: DataFrame with selected features that have AUC >= threshold
#     """
#     selected_features = []
#     auc_scores = []

#     if feature_cols is None:
#         feature_cols = X.columns.tolist()
#     # Binarize the output labels for multiclass ROC AUC calculation
#     y_bin=labelbinarizer(y)

#     # For multiclass, AUC is calculated in a One-vs-Rest fashion
#     for feature in feature_cols:
#         # Extract the feature column
#         X_feature = X[[feature]]
        
#         # Split data into train and test sets
#         X_train, X_test, y_train, y_test = train_test_split(X_feature, y, test_size=0.3, random_state=42)
        
#         # Train a simple classifier (RandomForest)
#         clf = RandomForestClassifier(n_estimators=100, random_state=42)
#         clf.fit(X_train, y_train)
        
#         # Get prediction probabilities
#         y_prob = clf.predict_proba(X_test)
        
#         # Compute the AUC-ROC score for each class in a One-vs-Rest approach
#         test_index = y_test.index
#         auc = roc_auc_score(y_bin[test_index], y_prob, multi_class='ovr')
        
#         auc_scores.append(auc)

#         # Select features with AUC >= threshold
#         if auc >= threshold:
#             selected_features.append(feature)
    
#     # Create a DataFrame with the results
#     results_df = pd.DataFrame({
#         'Feature': feature_cols,
#         'AUC': auc_scores
#     })

#     selected_df = results_df[results_df['AUC'] >= threshold].reset_index(drop=True)
#     if returnAUC:
#         return X[selected_df["Feature"]],selected_df["AUC"]
#     else:
#         return X[selected_df["Feature"]]
    
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, GroupShuffleSplit # Import GroupShuffleSplit
from sklearn.metrics import roc_auc_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelBinarizer # Import LabelBinarizer

def filterFeaturesByAUC(
    X,
    y,
    groups=None,
    feature_cols=None,
    threshold=0.580,
    returnAUC=False,
    classifier=LogisticRegression(solver='liblinear'),
    test_size=0.3,
    n_repeats=1
):
    """
    Calculate the AUC-ROC for each feature and select features with AUC >= threshold.
    Can handle binary or multi-class classification. Allows splitting based on groups.

    Parameters:
    X (pd.DataFrame): DataFrame with feature values
    y (pd.Series): Series with class labels
    groups (pd.Series, optional): Series with group labels for splitting.
    feature_cols (list, optional): List of column names that contain the feature values
    threshold (float): The AUC threshold for selecting predictive features (default is 0.580)
    returnAUC (bool): If True, return a tuple of (filtered_X, auc_series)
    classifier: The scikit-learn classifier to use for AUC calculation (default is LogisticRegression)
    test_size (float): The proportion of the dataset to include in the test split (default is 0.3)
    n_repeats (int): The number of times to repeat the split and AUC calculation (default is 1)

    Returns:
    pd.DataFrame: DataFrame with selected features that have AUC >= threshold
    pd.Series (optional): Series with the average AUC scores for the selected features if returnAUC is True
    """
    if feature_cols is None:
        feature_cols = X.columns.tolist()

    all_auc_scores = {feature: [] for feature in feature_cols}

    for repeat in range(n_repeats):
        # Split data into train and test sets, using groups if provided
        if groups is not None:
            splitter = GroupShuffleSplit(
                n_splits=1,
                test_size=test_size,
                random_state=42 + repeat # Vary random state for repeats
            )
            train_idx, test_idx = next(splitter.split(X, y, groups))
            X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
            y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
        else:
             # Use stratify if it's a classification problem and not multi-label
            stratify_y = y if (len(y.unique()) > 1 and not y.apply(type).eq(list).any()) else None
            X_train, X_test, y_train, y_test = train_test_split(
                X, y,
                test_size=test_size,
                random_state=42 + repeat, # Vary random state for repeats
                stratify=stratify_y
            )


        # Binarize the output labels for multiclass ROC AUC calculation if needed
        y_test_bin = y_test
        multi_class = 'raise' # Default to raise error for non-binary if not handled
        if len(y.unique()) > 2:
             lb = LabelBinarizer()
             y_test_bin = lb.fit_transform(y_test)
             multi_class = 'ovr'


        for feature in feature_cols:
            # Extract the feature column
            X_train_feature = X_train[[feature]]
            X_test_feature = X_test[[feature]]

            # Train the specified classifier
            clf = classifier
            clf.fit(X_train_feature, y_train)

            # Get prediction probabilities
            if hasattr(clf, 'predict_proba'):
                y_prob = clf.predict_proba(X_test_feature)
                # For binary, take the probability of the positive class
                if y_prob.shape[1] == 2:
                    y_prob = y_prob[:, 1]
                    # Ensure y_test is 1D
                    y_test_1d = y_test.values.ravel()
                    auc = roc_auc_score(y_test_1d, y_prob)
                else:
                     # For multi-class, calculate OvR AUC
                    auc = roc_auc_score(y_test_bin, y_prob, multi_class=multi_class)

            else:
                # If classifier doesn't have predict_proba, skip AUC for this feature
                print(f"Warning: Classifier {type(clf).__name__} does not support predict_proba. Skipping AUC for feature '{feature}'.")
                auc = 0.5 # Assign a neutral AUC if probabilities aren't available

            all_auc_scores[feature].append(auc)

    # Calculate average AUC over repeats
    average_auc_scores = {feature: np.mean(scores) for feature, scores in all_auc_scores.items()}

    selected_features = [feature for feature, avg_auc in average_auc_scores.items() if avg_auc >= threshold]

    filtered_X = X[selected_features]
    auc_series = pd.Series([average_auc_scores[feature] for feature in selected_features], index=selected_features)

    if returnAUC:
        return filtered_X, auc_series
    else:
        return filtered_X
        

def filterFeaturesByCorrelation(features, threshold=0.90,prognostic=None):
    """
    Remove highly correlated features (correlation coefficient ≥ 0.90).
    Retain the more prognostic feature from each correlated pair.
    """
    if prognostic is None:
        prognostic = np.ones(features.shape[1])
    corr_matrix = features.corr().abs()
    to_drop = set()    
    for i in range(len(corr_matrix.columns)):
        for j in range(i):
            if corr_matrix.iloc[i, j] >= threshold:
                feature_i = corr_matrix.columns[i]
                feature_j = corr_matrix.columns[j]
                # Here you would compare prognostic values and retain the better one
                if feature_j in to_drop:
                    continue
                if feature_j not in to_drop:  # Compare the features to determine which to drop
                    if prognostic[i] > prognostic[j]:
                        to_drop.add(feature_j)
                    else:
                        to_drop.add(feature_i)
    for a in to_drop:
        print(f"Feature {a} is highly correlated and will be removed")
              
    return features.drop(columns=to_drop)



def create_test_data(num_samples=100, num_features=10):
    # Create a DataFrame with random feature values
    features = pd.DataFrame(np.random.rand(num_samples, num_features), columns=[f'feature_{i}' for i in range(num_features)])
    # Create a Series with random target values
    targets = pd.Series(np.random.randint(0, 3, num_samples))
    return features, targets

if __name__ == "__main__":
    
    features, targets = create_test_data()
    features.iloc[:,1]=features.iloc[:,0]+2*features.iloc[:,0]
    features,resultDF=filterFeaturesByAUC(features, targets, feature_cols=None, threshold=0.5)
    features=filterFeaturesByCorrelation(features, threshold=0.44,prognostic=resultDF["AUC"].values)
    print(features.columns)