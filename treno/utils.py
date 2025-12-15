import torch
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, roc_auc_score, multilabel_confusion_matrix
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import StandardScaler, LabelBinarizer

# Helper functions for confusion matrix metrics (previously in pynico.stats)
def accuracyFromConfusion(c):
    tn, fp, fn, tp = c.ravel()
    return (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0

def specificityFromConfusion(c):
    tn, fp, fn, tp = c.ravel()
    return tn / (tn + fp) if (tn + fp) > 0 else 0

def sensitivityFromConfusion(c):
    tn, fp, fn, tp = c.ravel()
    return tp / (tp + fn) if (tp + fn) > 0 else 0
from sklearn.model_selection import train_test_split, GroupShuffleSplit, StratifiedGroupKFold
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.base import is_classifier, is_regressor
from sklearn.metrics import roc_auc_score, r2_score
from scipy.stats import pearsonr
from scipy.signal import convolve2d

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
        o = {"accuracy": accuracyFromConfusion(c),
             "specificity": specificityFromConfusion(c),
             "sensitivity": sensitivityFromConfusion(c),
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



from sklearn.ensemble import RandomForestClassifier

def feature_selection(
    features,
    targets,
    groups=None,
    score_threshold=0.5,
    corr_threshold=0.9,
    score_model=None,
    score_test_size=0.3,
    score_n_repeats=1,
    gini_n_repeats=10,
    gini_test_size=0.1,
    gini_random_seed=None,
    return_gini=False,
    task="classification"
):
    """
    Feature selection pipeline for classification or regression.

    Parameters:
        features: DataFrame or ndarray of features
        targets: Series or ndarray of targets
        groups: Optional, group labels for splitting
        score_threshold: Threshold for feature score selection
        corr_threshold: Correlation threshold for feature removal
        score_model: scikit-learn model to use for scoring (default: RandomForestClassifier or RandomForestRegressor)
        score_test_size: Test size for score evaluation
        score_n_repeats: Number of repeats for score evaluation
        gini_n_repeats: Number of repeats for Gini ranking
        gini_test_size: Test size for Gini ranking
        gini_random_seed: Random seed for Gini ranking
        return_gini: If True, also return Gini scores
        task: "classification" or "regression"

    Returns:
        DataFrame of selected features (optionally sorted by Gini), and optionally Gini scores
    """
    # 1. Remove NaNs
    features, targets = remove_nans(features, targets)

    # 2. Normalize features
    # features = zScoreFeatures(features)

    # 3. Filter by MAD
    features = filterFeaturesByMAD(features)
    if features.shape[1] == 0:
        raise ValueError("No features passed MAD filter")

    # 4. Filter by score (user-selected model or default)
    if score_model is not None:
        model = score_model
    else:
        model = RandomForestClassifier() if task == "classification" else RandomForestRegressor()
    features, score_values = filterFeaturesByScore(
        features,
        targets,
        groups=groups,
        threshold=score_threshold,
        return_score=True,
        model=model,
        test_size=score_test_size,
        n_repeats=score_n_repeats
    )
    if features.shape[1] == 0:
        raise ValueError("No features passed score filter")

    # 5. Filter by correlation
    features = filterFeaturesByCorrelation(
        features,
        threshold=corr_threshold,
        score=score_values.values
    )
    if features.shape[1] == 0:
        raise ValueError("No features passed correlation filter")

    # 6. Rank by repeated Gini index (optional)
    if return_gini:
        features_sorted, gini_ranks = rankFeaturesByRepeatedGini(
            features, targets,
            n_repeats=gini_n_repeats,
            test_size=gini_test_size,
            random_seed=gini_random_seed,
            groups=groups,
            return_gini=True
        )
        return features_sorted, gini_ranks
    else:
        features_sorted = rankFeaturesByRepeatedGini(
            features, targets,
            n_repeats=gini_n_repeats,
            test_size=gini_test_size,
            random_seed=gini_random_seed,
            groups=groups,
            return_gini=False
        )
        return features_sorted


# ============================================================================
# Data Splitting Utilities (Medical Imaging Aware)
# ============================================================================

def extract_patient_groups(dataframe_index, augmentation_suffix='-aug'):
    """
    Extract patient groups from DataFrame index, handling data augmentation.
    
    Critical for medical imaging: Ensures augmented samples from the same patient
    stay in the same split (train or test), preventing data leakage.
    
    Parameters:
        dataframe_index: DataFrame index (can contain augmentation suffixes)
        augmentation_suffix: Suffix used to mark augmented samples
        
    Returns:
        list: Group labels for each sample
        
    Example:
        >>> index = ['patient_001', 'patient_001-aug', 'patient_002', 'patient_002-aug']
        >>> groups = extract_patient_groups(index)
        >>> # groups = [0, 0, 1, 1]  # Same patient = same group
    """
    # Get unique patients (without augmentation suffix)
    unique_patients = [idx for idx in dataframe_index if augmentation_suffix not in str(idx)]
    
    groups = []
    for idx in dataframe_index:
        # Remove augmentation suffix to get base patient ID
        base_idx = str(idx).split(augmentation_suffix)[0]
        
        # Find group number
        if base_idx in unique_patients:
            group_num = unique_patients.index(base_idx)
        else:
            # If not found, add it (shouldn't happen normally)
            unique_patients.append(base_idx)
            group_num = len(unique_patients) - 1
            
        groups.append(group_num)
    
    return groups


def stratified_group_split(X, y, groups=None, test_size=0.25, random_state=None, 
                          augmentation_suffix='-aug'):
    """
    Stratified train/test split respecting patient groups.
    
    **Critical for medical imaging**: Ensures that:
    1. Patients don't leak between train/test sets
    2. Augmented samples stay with their original patient
    3. Class distribution is preserved (stratification)
    
    Parameters:
        X: Features (DataFrame or array)
        y: Labels (DataFrame or array)
        groups: Group labels (if None, extracted from X.index)
        test_size: Fraction of data for testing
        random_state: Random seed for reproducibility
        augmentation_suffix: Suffix marking augmented samples
        
    Returns:
        X_train, X_test, y_train, y_test, groups_train, groups_test
        
    Example:
        >>> # DataFrame with patient IDs as index
        >>> X_train, X_test, y_train, y_test, g_train, g_test = stratified_group_split(
        ...     X, y, test_size=0.25, random_state=42
        ... )
        >>> # No patient appears in both train and test!
    """
    # Convert to DataFrame if needed
    if not isinstance(X, pd.DataFrame):
        X = pd.DataFrame(X)
    if not isinstance(y, (pd.DataFrame, pd.Series)):
        y = pd.Series(y)
    
    # Extract groups if not provided
    if groups is None:
        groups = extract_patient_groups(X.index, augmentation_suffix)
    groups = np.array(groups)
    
    # Determine number of splits
    n_splits = int(1 / test_size)
    
    # Use StratifiedGroupKFold
    sgkf = StratifiedGroupKFold(n_splits=n_splits, random_state=random_state, shuffle=True)
    
    # Get first split
    train_idx, test_idx = next(sgkf.split(X, y, groups))
    
    # Return splits
    if isinstance(X, pd.DataFrame):
        return (X.iloc[train_idx], X.iloc[test_idx], 
                y.iloc[train_idx], y.iloc[test_idx],
                groups[train_idx], groups[test_idx])
    else:
        return (X[train_idx], X[test_idx],
                y[train_idx], y[test_idx],
                groups[train_idx], groups[test_idx])


# ============================================================================
# Additional Evaluation and Visualization Utilities
# ============================================================================

def compute_metrics(all_labels, all_preds, multilabel=False):
    """
    Compute comprehensive classification metrics.
    
    Parameters:
        all_labels (array-like): Ground truth labels
        all_preds (array-like): Predicted labels
        multilabel (bool): Whether this is multi-label classification
    
    Returns:
        dict: Dictionary containing accuracy, precision, recall, f1, and confusion matrix
        
    Example:
        >>> metrics = compute_metrics(y_true, y_pred, multilabel=False)
        >>> print(f"Accuracy: {metrics['accuracy']:.3f}")
        >>> print(f"F1 Score: {metrics['f1']:.3f}")
    """
    if multilabel:
        accuracy = (all_preds == all_labels).mean()
        precision = precision_score(all_labels, all_preds, average="micro", zero_division=0)
        recall = recall_score(all_labels, all_preds, average="micro", zero_division=0)
        f1 = f1_score(all_labels, all_preds, average="micro", zero_division=0)
    else:
        accuracy = accuracy_score(all_labels, all_preds)
        precision = precision_score(all_labels, all_preds, average="weighted", zero_division=0)
        recall = recall_score(all_labels, all_preds, average="weighted", zero_division=0)
        f1 = f1_score(all_labels, all_preds, average="weighted", zero_division=0)

    cm = confusion_matrix(all_labels, all_preds)
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'confusion_matrix': cm
    }


def compute_binary_metrics(y_true, y_pred, eps=1e-6):
    """
    Compute comprehensive binary classification metrics including clinical measures.
    
    Parameters:
        y_true (array-like): Ground truth binary labels
        y_pred (array-like): Predicted binary labels
        eps (float): Small constant to avoid division by zero
        
    Returns:
        dict: Comprehensive metrics including sensitivity, specificity, MCC, odds ratio, etc.
        
    Example:
        >>> metrics = compute_binary_metrics(y_true, y_pred)
        >>> print(f"Sensitivity: {metrics['sensitivity']:.3f}")
        >>> print(f"Odds Ratio: {metrics['odds_ratio']:.3f}")
        >>> print(f"MCC: {metrics['mcc']:.3f}")
    """
    try:
        from sklearn import metrics as sk_metrics
        from scipy.stats.contingency import relative_risk, odds_ratio
    except ImportError:
        raise ImportError("scipy and sklearn required for compute_binary_metrics")
    
    y_true = np.asarray(y_true).flatten()
    y_pred = np.asarray(y_pred).flatten()
    
    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred).astype(np.float32)
    n = cm.sum()
    
    # Extract values (note: sklearn confusion matrix is [[TN, FP], [FN, TP]])
    tn, fp, fn, tp = cm.ravel()
    
    # Basic metrics
    accuracy = (tp + tn) / n
    precision = tp / (tp + fp + eps)
    recall = tp / (tp + fn + eps)
    f1 = 2 * (precision * recall) / (precision + recall + eps)
    specificity = tn / (tn + fp + eps)
    sensitivity = tp / (tp + fn + eps)
    
    # Matthews Correlation Coefficient
    mcc = (tp * tn - fp * fn) / (
        np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)) + eps
    )
    
    # Error rate
    error_rate = (fp + fn) / n
    
    # Relative risk and odds ratio (handle edge cases)
    try:
        rr = relative_risk(*cm.astype(int).ravel()).relative_risk
        if np.isinf(rr):
            rr = np.nan
    except:
        rr = np.nan
    
    try:
        or_val = odds_ratio(cm.astype(int)).statistic
        if np.isinf(or_val):
            or_val = np.nan
    except:
        or_val = np.nan
    
    # AUC and optimal threshold
    try:
        auc = sk_metrics.roc_auc_score(y_true, y_pred)
        fpr, tpr, thresholds = sk_metrics.roc_curve(y_true, y_pred)
        optimal_idx = np.argmax(tpr - fpr)
        auc_threshold = thresholds[optimal_idx]
    except:
        auc = np.nan
        auc_threshold = np.nan
    
    return {
        "true_negatives": tn / n,
        "true_positives": tp / n,
        "false_negatives": fn / n,
        "false_positives": fp / n,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "specificity": specificity,
        "sensitivity": sensitivity,
        "mcc": mcc,
        "error_rate": error_rate,
        "relative_risk": rr,
        "odds_ratio": or_val,
        "auc": auc,
        "auc_threshold": auc_threshold,
    }


def compute_multilabel_sensitivity_specificity(cm):
    """
    Compute sensitivity and specificity for each class from confusion matrix.
    
    Parameters:
        cm (np.ndarray): Confusion matrix (2D numpy array)
    
    Returns:
        tuple: (sensitivities, specificities) - lists for each class
        
    Example:
        >>> cm = confusion_matrix(y_true, y_pred)
        >>> sens, spec = compute_multilabel_sensitivity_specificity(cm)
        >>> for i, (s, sp) in enumerate(zip(sens, spec)):
        ...     print(f"Class {i}: Sensitivity={s:.3f}, Specificity={sp:.3f}")
    """
    num_classes = cm.shape[0]
    sensitivities = []
    specificities = []

    for i in range(num_classes):
        tp = cm[i, i]
        fn = np.sum(cm[i, :]) - tp
        fp = np.sum(cm[:, i]) - tp
        tn = np.sum(cm) - (tp + fn + fp)

        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0

        sensitivities.append(sensitivity)
        specificities.append(specificity)

    return sensitivities, specificities


def visualize_embeddings(feature_vectors, labels, method="tsne", save_path=None):
    """
    Visualize feature vectors using dimensionality reduction.
    
    Parameters:
        feature_vectors (list or np.ndarray): Extracted features from model
        labels (list or np.ndarray): Corresponding labels for each sample
        method (str): "tsne" or "pca" for dimensionality reduction
        save_path (str, optional): Path to save the plot
        
    Example:
        >>> # Extract features from your model
        >>> features = []
        >>> labels = []
        >>> for batch_x, batch_y in dataloader:
        ...     with torch.no_grad():
        ...         feat = model.extract_features(batch_x)
        ...     features.append(feat.cpu().numpy())
        ...     labels.append(batch_y.numpy())
        >>> visualize_embeddings(features, labels, method="tsne")
    """
    try:
        from sklearn.manifold import TSNE
        from sklearn.decomposition import PCA
        import matplotlib.pyplot as plt
    except ImportError:
        print("Warning: sklearn and matplotlib required for visualize_embeddings")
        return

    feature_vectors = np.vstack(feature_vectors)
    labels = np.concatenate(labels) if isinstance(labels[0], np.ndarray) else np.array(labels)

    if method == "tsne":
        reducer = TSNE(n_components=2, perplexity=min(30, len(feature_vectors) - 1), random_state=42)
    else:
        reducer = PCA(n_components=2)

    embedded_features = reducer.fit_transform(feature_vectors)

    plt.figure(figsize=(8, 6))
    scatter = plt.scatter(embedded_features[:, 0], embedded_features[:, 1], 
                         c=labels, cmap="coolwarm", alpha=0.7)
    plt.colorbar(scatter, label="Class")
    plt.title(f"{method.upper()} Visualization of Feature Embeddings")
    plt.xlabel(f"{method.upper()} Component 1")
    plt.ylabel(f"{method.upper()} Component 2")
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()


def write_confusion_matrix_to_tensorboard(writer, cm, min_val, max_val, tag="cm", epoch=0, colormap=None):
    """
    Write a confusion matrix to TensorBoard as a colored image.
    
    Parameters:
        writer: TensorBoard SummaryWriter object
        cm (np.ndarray): Confusion matrix (2D array)
        min_val (float): Minimum value for normalization
        max_val (float): Maximum value for normalization
        tag (str): Tag for the image in TensorBoard
        epoch (int): Epoch number (step in TensorBoard)
        colormap: Matplotlib colormap (default: viridis)
        
    Example:
        >>> from torch.utils.tensorboard import SummaryWriter
        >>> writer = SummaryWriter('runs/experiment')
        >>> cm = confusion_matrix(y_true, y_pred)
        >>> write_confusion_matrix_to_tensorboard(writer, cm, 0, cm.max(), epoch=10)
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("Warning: matplotlib required for TensorBoard confusion matrix visualization")
        return
    
    if colormap is None:
        colormap = plt.cm.viridis
    
    # Normalize
    cm_normalized = (cm - min_val) / (max_val - min_val)
    cm_normalized = np.clip(cm_normalized, 0, 1)

    # Apply colormap
    cm_colored = colormap(cm_normalized)
    cm_colored = cm_colored[:, :, :3]  # Remove alpha channel

    # Convert to tensor
    cm_tensor = torch.tensor(cm_colored, dtype=torch.float32)
    cm_tensor = cm_tensor.permute(2, 0, 1)  # CHW format

    writer.add_image(tag, cm_tensor, epoch, dataformats="CHW")


# ============================================================================
# Explainability Utilities (Grad-CAM & Saliency Maps)
# ============================================================================

class GradCAM:
    """
    Gradient-weighted Class Activation Mapping (Grad-CAM) for CNNs.
    
    Example:
        >>> model = EMUNet(in_channels=1, out_channels=3)
        >>> gradcam = GradCAM(model, target_layer=model.encoder[-1])
        >>> 
        >>> # Forward pass with target class
        >>> cam = gradcam(input_tensor, target_class=1)
        >>> 
        >>> # Upsample to input size
        >>> cam_upsampled = gradcam.upsample_cam(cam, input_tensor.shape[-3:])
    """
    def __init__(self, model, target_layer):
        """
        Initialize Grad-CAM.
        
        Parameters:
            model: PyTorch model
            target_layer: The layer to compute CAM from (typically last conv layer)
        """
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        
        # Register hooks
        self.forward_handle = target_layer.register_forward_hook(self._forward_hook)
        self.backward_handle = target_layer.register_full_backward_hook(self._backward_hook)
    
    def _forward_hook(self, module, input, output):
        """Capture forward activations."""
        self.activations = output.detach()
    
    def _backward_hook(self, module, grad_input, grad_output):
        """Capture backward gradients."""
        self.gradients = grad_output[0].detach()
    
    def __call__(self, input_tensor, target_class=None, normalize=True):
        """
        Generate Grad-CAM heatmap.
        
        Parameters:
            input_tensor: Input tensor (B, C, D, H, W) or (B, C, H, W)
            target_class: Target class index (if None, uses predicted class)
            normalize: Whether to normalize CAM to [0, 1]
            
        Returns:
            cam: Grad-CAM heatmap (D, H, W) or (H, W)
        """
        self.model.eval()
        input_tensor.requires_grad = True
        
        # Forward pass
        self.model.zero_grad()
        output = self.model(input_tensor)
        
        # Get target class
        if target_class is None:
            target_class = output.argmax(dim=1).item()
        
        # Backward pass
        score = output[0, target_class]
        score.backward()
        
        # Compute CAM
        if self.gradients is None or self.activations is None:
            raise RuntimeError("Gradients or activations not captured. Check target layer.")
        
        # Global average pooling of gradients
        weights = self.gradients.mean(dim=tuple(range(2, self.gradients.ndim)))  # (B, C)
        
        # Weighted combination of activation maps
        cam = torch.zeros(self.activations.shape[2:], device=self.activations.device)
        for i, w in enumerate(weights[0]):
            cam += w * self.activations[0, i]
        
        # Apply ReLU (only positive influences)
        cam = torch.relu(cam)
        
        # Normalize
        if normalize and cam.max() > 0:
            cam = cam / cam.max()
        
        return cam.cpu().numpy()
    
    def upsample_cam(self, cam, target_size, mode='trilinear'):
        """
        Upsample CAM to target size.
        
        Parameters:
            cam: CAM array (D, H, W) or (H, W)
            target_size: Target spatial dimensions
            mode: Interpolation mode ('trilinear' for 3D, 'bilinear' for 2D)
            
        Returns:
            Upsampled CAM as numpy array
        """
        import torch.nn.functional as F
        
        cam_tensor = torch.from_numpy(cam).unsqueeze(0).unsqueeze(0)  # (1, 1, ...)
        
        if len(target_size) == 3:
            upsampled = F.interpolate(cam_tensor, size=target_size, 
                                     mode='trilinear', align_corners=False)
        elif len(target_size) == 2:
            upsampled = F.interpolate(cam_tensor, size=target_size, 
                                     mode='bilinear', align_corners=False)
        else:
            raise ValueError(f"Unsupported target size: {target_size}")
        
        return upsampled.squeeze().numpy()
    
    def remove_hooks(self):
        """Remove registered hooks."""
        self.forward_handle.remove()
        self.backward_handle.remove()
    
    def __del__(self):
        """Cleanup hooks on deletion."""
        try:
            self.remove_hooks()
        except:
            pass


def compute_saliency_map(model, input_tensor, target_class=None, smooth=True, smooth_size=3):
    """
    Compute saliency map using input gradients.
    
    Parameters:
        model: PyTorch model
        input_tensor: Input tensor (B, C, D, H, W) or (B, C, H, W)
        target_class: Target class index (if None, uses predicted class)
        smooth: Whether to apply smoothing filter
        smooth_size: Size of smoothing kernel
        
    Returns:
        saliency: Saliency map as numpy array
        
    Example:
        >>> model = EMUNet(in_channels=1, out_channels=3)
        >>> saliency = compute_saliency_map(model, input_tensor, target_class=1)
        >>> # Apply mask to focus on brain region
        >>> saliency[brain_mask == 0] = 0
    """
    try:
        from scipy.ndimage import uniform_filter
    except ImportError:
        uniform_filter = None
    
    model.eval()
    input_tensor.requires_grad = True
    
    # Forward pass
    model.zero_grad()
    output = model(input_tensor)
    
    # Get target class
    if target_class is None:
        target_class = output.argmax(dim=1).item()
    
    # Backward pass
    score = output[0, target_class]
    score.backward()
    
    # Get gradients
    gradients = input_tensor.grad.cpu().detach().numpy()
    
    # Take absolute value and remove batch/channel dims
    saliency = np.abs(gradients[0, 0])  # Assuming single channel
    
    # Smooth if requested
    if smooth and uniform_filter is not None:
        saliency = uniform_filter(saliency, size=smooth_size)
    
    return saliency


def postprocess_cam(cam, mask=None, smooth=True, smooth_size=3, normalize=True):
    """
    Post-process CAM/saliency maps with smoothing, masking, and normalization.
    
    Parameters:
        cam: CAM or saliency array
        mask: Binary mask to apply (e.g., brain mask)
        smooth: Whether to apply smoothing
        smooth_size: Smoothing kernel size
        normalize: Whether to normalize to [0, 1]
        
    Returns:
        Processed CAM array
        
    Example:
        >>> cam = gradcam(input_tensor, target_class=1)
        >>> cam_upsampled = gradcam.upsample_cam(cam, input_tensor.shape[-3:])
        >>> cam_final = postprocess_cam(cam_upsampled, mask=brain_mask, smooth=True)
    """
    cam_processed = cam.copy()
    
    # Smooth BEFORE masking to avoid edge artifacts
    if smooth:
        try:
            from scipy.ndimage import uniform_filter
            cam_processed = uniform_filter(cam_processed, size=smooth_size)
        except ImportError:
            pass
    
    # Normalize BEFORE masking so relative values are preserved
    if normalize and cam_processed.max() > 0:
        cam_processed = (cam_processed - cam_processed.min()) / (cam_processed.max() - cam_processed.min())
    
    # Apply mask LAST to preserve relative intensities
    if mask is not None:
        cam_processed = cam_processed * mask
    
    return cam_processed


# ============================================================================
# Medical Imaging Utilities
# ============================================================================

def resize_image(arr, target_size):
    """
    Crop or pad a 2D or 3D array to the target size.
    
    Parameters:
        arr (np.ndarray): Input array (2D or 3D)
        target_size (tuple): Target size (H, W) for 2D or (H, W, D) for 3D
        
    Returns:
        np.ndarray: Cropped or padded array
        
    Example:
        >>> img = np.random.rand(100, 100, 50)
        >>> resized = resize_image(img, (128, 128, 64))
        >>> print(resized.shape)  # (128, 128, 64)
    """
    arr = np.asarray(arr)
    current_size = arr.shape
    padded_array = np.zeros(target_size, dtype=arr.dtype)
    
    def crop_or_pad_dims(curr, targ):
        return max((curr - targ) // 2, 0), max((targ - curr) // 2, 0)
    
    if len(current_size) == 2:
        crop_y, pad_y = crop_or_pad_dims(current_size[0], target_size[0])
        crop_x, pad_x = crop_or_pad_dims(current_size[1], target_size[1])
        
        cropped = arr[crop_y:crop_y + min(current_size[0], target_size[0]),
                      crop_x:crop_x + min(current_size[1], target_size[1])]
        padded_array[pad_y:pad_y + cropped.shape[0], pad_x:pad_x + cropped.shape[1]] = cropped

    elif len(current_size) == 3:
        crop_y, pad_y = crop_or_pad_dims(current_size[0], target_size[0])
        crop_x, pad_x = crop_or_pad_dims(current_size[1], target_size[1])
        crop_depth, pad_depth = crop_or_pad_dims(current_size[2], target_size[2])
        
        cropped = arr[crop_y:crop_y + min(current_size[0], target_size[0]),
                      crop_x:crop_x + min(current_size[1], target_size[1]), 
                      crop_depth:crop_depth + min(current_size[2], target_size[2])]
        padded_array[pad_y:pad_y + cropped.shape[0], 
                     pad_x:pad_x + cropped.shape[1], 
                     pad_depth:pad_depth + cropped.shape[2]] = cropped
    else:
        raise ValueError("Input array must be 2D or 3D.")
    
    return padded_array


def store_3d_array(im, image_data):
    """
    Extract a block from source 3D array that maximizes nonzero pixels.
    Uses intelligent cropping to select the most informative region.
    
    Parameters:
        im (np.ndarray): Source 3D array of shape (src_h, src_w, src_d)
        image_data (np.ndarray): Target 3D array defining desired output shape
            
    Returns:
        np.ndarray: Array of shape image_data.shape containing selected data
        
    Example:
        >>> source = np.random.rand(200, 200, 100)
        >>> target = np.zeros((128, 128, 64))
        >>> result = store_3d_array(source, target)
        >>> print(result.shape)  # (128, 128, 64)
    """
    target_h, target_w, target_d = image_data.shape
    src_h, src_w, src_d = im.shape

    # Handle depth dimension
    if src_d >= target_d:
        im_eff = im[:, :, src_d - target_d:]
        d_eff = target_d
    else:
        im_eff = im.copy()
        d_eff = src_d

    # Extract spatial block with most nonzero pixels
    if im_eff.shape[0] >= target_h and im_eff.shape[1] >= target_w:
        mask = (im_eff != 0).astype(np.int32)
        mask2d = mask.sum(axis=2)
        
        kernel = np.ones((target_h, target_w), dtype=np.int32)
        conv_result = convolve2d(mask2d, kernel, mode='valid')
        
        i0, j0 = np.unravel_index(np.argmax(conv_result), conv_result.shape)
        block = im_eff[i0:i0+target_h, j0:j0+target_w, :d_eff]
    else:
        block = np.zeros((target_h, target_w, d_eff), dtype=im.dtype)
        copy_h = min(im_eff.shape[0], target_h)
        copy_w = min(im_eff.shape[1], target_w)
        offset_h = (target_h - copy_h) // 2
        offset_w = (target_w - copy_w) // 2
        block[offset_h:offset_h+copy_h, offset_w:offset_w+copy_w, :] = im_eff[:copy_h, :copy_w, :d_eff]
    
    # Ensure correct depth
    if d_eff < target_d:
        final_block = np.zeros((target_h, target_w, target_d), dtype=im.dtype)
        final_block[:, :, :d_eff] = block
    else:
        final_block = block

    return final_block
    
    
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