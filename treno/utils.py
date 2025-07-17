import torch
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, roc_auc_score, multilabel_confusion_matrix
from pynico_eros_montin import stats as st
from sklearn.preprocessing import StandardScaler, LabelBinarizer
from sklearn.model_selection import train_test_split, GroupShuffleSplit
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.base import is_classifier, is_regressor
from sklearn.metrics import roc_auc_score, r2_score
from scipy.stats import pearsonr

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

def remove_nans(FEATURES, LABELS):
    """
    Drop rows with any NaNs in FEATURES or LABELS, keeping common indices.
    Works for both pandas DataFrame/Series and numpy arrays.
    """
    # Convert to pandas if numpy
    if isinstance(FEATURES, np.ndarray):
        FEATURES = pd.DataFrame(FEATURES)
    if isinstance(LABELS, np.ndarray):
        LABELS = pd.Series(LABELS)
    f = FEATURES.dropna()
    l = LABELS.dropna()
    idx = f.index.intersection(l.index)
    return f.loc[idx], l.loc[idx]

def zScoreFeatures(features):
    """Standardize features using Z-score normalization."""
    if isinstance(features, np.ndarray):
        features = pd.DataFrame(features)
    scaler = StandardScaler()
    features = pd.DataFrame(scaler.fit_transform(features), columns=features.columns)
    return features

def filterFeaturesByMAD(features):
    """
    Calculate Median Absolute Deviation (MAD) for each feature.
    Discard features with MAD equal to zero.
    """
    if isinstance(features, np.ndarray):
        features = pd.DataFrame(features)
    mad_values = calculate_df_mad(features)
    return features.loc[:, mad_values != 0]

def calculate_df_mad(df):
    """
    Calculate the Median Absolute Deviation (MAD) for each column of the DataFrame.
    """
    if isinstance(df, np.ndarray):
        df = pd.DataFrame(df)
    mad = df.apply(lambda x: np.median(np.abs(x - np.median(x))), axis=0)
    return mad

def labelbinarizer(y):
    if isinstance(y, np.ndarray):
        y = pd.Series(y)
    lb = LabelBinarizer()
    y_bin = lb.fit_transform(y)
    return y_bin

def gini_index(x, y):
    x = np.asarray(x)
    y = np.asarray(y)
    order = np.argsort(x)
    y_sorted = y[order]
    n = len(y)
    cum_y = np.cumsum(y_sorted)
    if cum_y[-1] == 0:
        return 0
    gini = (np.sum(cum_y) / (cum_y[-1] * n)) - (n + 1) / (2 * n)
    return gini

def rankFeaturesByRepeatedGini(
    X,
    y,
    n_repeats: int = 10,
    test_size: float = 0.1,
    random_seed: int = None,
    groups: pd.Series = None,
    return_gini: bool = False
):
    if isinstance(X, np.ndarray):
        X = pd.DataFrame(X)
    if isinstance(y, np.ndarray):
        y = pd.Series(y)
    feature_cols = X.columns.tolist()
    gini_scores = {feature: [] for feature in feature_cols}

    for i in range(n_repeats):
        rs = (random_seed or 0) + i
        if groups is not None:
            splitter = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=rs)
            train_idx, _ = next(splitter.split(X, y, groups))
            X_tr, y_tr = X.iloc[train_idx], y.iloc[train_idx]
        else:
            X_tr, _, y_tr, _ = train_test_split(
                X, y, test_size=test_size, random_state=rs, stratify=y if len(np.unique(y)) > 1 else None
            )
        for feature in feature_cols:
            try:
                g = gini_index(X_tr[feature].values, y_tr.values)
            except Exception:
                g = 0
            gini_scores[feature].append(g)

    avg_gini = {feature: np.mean(scores) for feature, scores in gini_scores.items()}
    ranked = pd.Series(avg_gini).sort_values(ascending=False)
    X_sorted = X[ranked.index]

    if return_gini:
        return X_sorted, ranked
    else:
        return X_sorted

def filterFeaturesByScore(
    X,
    y,
    groups=None,
    feature_cols=None,
    threshold=0.580,
    return_score=False,
    model=None,
    test_size=0.3,
    n_repeats=1,
    return_all_scores=False
):
    if isinstance(X, np.ndarray):
        X = pd.DataFrame(X)
    if isinstance(y, np.ndarray):
        y = pd.Series(y)
    if feature_cols is None:
        feature_cols = X.columns.tolist()

    # If no model is provided, use RandomForestClassifier for classification, RandomForestRegressor for regression
    is_reg = np.issubdtype(y.dtype, np.floating) or (len(np.unique(y)) > 10 and y.dtype != object)
    if model is None:
        if is_reg:
            model = RandomForestRegressor()
        else:
            model = RandomForestClassifier()

    all_scores = {feature: [] for feature in feature_cols}

    for repeat in range(n_repeats):
        # Split data into train and test sets, using groups if provided
        if groups is not None:
            splitter = GroupShuffleSplit(
                n_splits=1,
                test_size=test_size,
                random_state=42 + repeat
            )
            train_idx, test_idx = next(splitter.split(X, y, groups))
            X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
            y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
        else:
            is_reg = is_regressor(model)
            stratify_y = y if (not is_reg and len(y.unique()) > 1 and not y.apply(type).eq(list).any()) else None
            X_train, X_test, y_train, y_test = train_test_split(
                X, y,
                test_size=test_size,
                random_state=42 + repeat,
                stratify=stratify_y
            )

        X_train = zScoreFeatures(X_train)
        X_test = zScoreFeatures(X_test)

        for feature in feature_cols:
            X_train_feature = X_train[[feature]]
            X_test_feature = X_test[[feature]]

            # Classification
            if is_classifier(model):
                clf = model
                clf.fit(X_train_feature, y_train)
                if hasattr(clf, 'predict_proba'):
                    y_prob = clf.predict_proba(X_test_feature)
                    if y_prob.shape[1] == 2:
                        y_prob = y_prob[:, 1]
                        y_test_1d = y_test.values.ravel()
                        if len(np.unique(y_test_1d)) < 2:
                            score = 0.5
                        else:
                            score = roc_auc_score(y_test_1d, y_prob)
                    else:
                        lb = LabelBinarizer()
                        y_test_bin = lb.fit_transform(y_test)
                        if y_test_bin.shape[1] > 1 and len(np.unique(y_test_bin[:, np.argmax(y_prob, axis=1)])) > 1:
                            score = roc_auc_score(y_test_bin, y_prob, multi_class='ovr')
                        else:
                            score = 0.5
                else:
                    print(f"Warning: Classifier {type(clf).__name__} does not support predict_proba. Skipping score for feature '{feature}'.")
                    score = 0.5
            # Regression
            elif is_regressor(model):
                reg = model
                reg.fit(X_train_feature, y_train)
                y_pred = reg.predict(X_test_feature)
                try:
                    score = abs(pearsonr(y_test.values.ravel(), y_pred.ravel())[0])
                    if np.isnan(score):
                        score = 0
                except Exception:
                    score = 0
            else:
                raise ValueError("Model must be a scikit-learn classifier or regressor.")

            all_scores[feature].append(score)

    average_scores = {feature: np.mean(scores) for feature, scores in all_scores.items()}
    score_series_all = pd.Series(average_scores).sort_values(ascending=False)
    selected_features = score_series_all[score_series_all >= threshold].index.tolist()
    filtered_X = X[selected_features]
    score_series_selected = score_series_all.loc[selected_features]

    if return_all_scores:
        return score_series_all
    elif return_score:
        return filtered_X, score_series_selected
    else:
        return filtered_X

def filterFeaturesByCorrelation(features, threshold=0.90, score=None):
    if isinstance(features, np.ndarray):
        features = pd.DataFrame(features)
    if score is None:
        score = np.ones(features.shape[1])
    corr_matrix = features.corr().abs()
    to_drop = set()    
    for i in range(len(corr_matrix.columns)):
        for j in range(i):
            if corr_matrix.iloc[i, j] >= threshold:
                feature_i = corr_matrix.columns[i]
                feature_j = corr_matrix.columns[j]
                if feature_j in to_drop:
                    continue
                if score[i] > score[j]:
                    to_drop.add(feature_j)
                else:
                    to_drop.add(feature_i)
    for a in to_drop:
        print(f"Feature {a} is highly correlated and will be removed")
    return features.drop(columns=to_drop)

def create_test_data(num_samples=100, num_features=10):
    features = pd.DataFrame(np.random.rand(num_samples, num_features), columns=[f'feature_{i}' for i in range(num_features)])
    targets = pd.Series(np.random.randint(0, 3, num_samples))
    return features, targets

def generate_fake_data(n_samples=100, n_features=20, n_groups=5, classification=True, random_state=None):
    """
    Generates fake data for testing feature selection and classification.
    """
    if random_state is not None:
        np.random.seed(random_state)
    x = pd.DataFrame(np.random.rand(n_samples, n_features), columns=[f'feature_{i}' for i in range(n_features)])
    if classification:
        y = pd.Series(np.random.randint(0, 2, n_samples), name='label')
    else:
        weights = np.random.rand(n_features)
        y = pd.Series(np.dot(x, weights) + np.random.randn(n_samples) * 0.5, name='label')
    groups = None
    if n_groups and n_groups > 0:
        groups = pd.Series(np.random.randint(0, n_groups, n_samples), name='group')
    return x, y, groups

if __name__ == "__main__":
    # Example usage and demonstration
    features, targets = create_test_data()
    features.iloc[:,1] = features.iloc[:,0] + 2 * features.iloc[:,0]

    # 1. Feature selection by score (keeps only selected features)
    features_selected = filterFeaturesByScore(features, targets, feature_cols=None, threshold=0.5)
    print("Features after score filtering:", features_selected.columns.tolist())

    # 2. Filter by correlation (keeps only remaining features)
    features_final = filterFeaturesByCorrelation(features_selected, threshold=0.44)
    print("Features after correlation filter:", features_final.columns.tolist())

    # 3. Gini ranking (keeps only remaining features, sorted)
    features_gini_sorted = rankFeaturesByRepeatedGini(features_final, targets, n_repeats=10)
    print("Features after Gini ranking (sorted):", features_gini_sorted.columns.tolist())