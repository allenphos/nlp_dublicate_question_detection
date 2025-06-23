import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.metrics import f1_score, log_loss, confusion_matrix, classification_report
from xgboost import XGBClassifier
import joblib

class ClassicMLTrainer:
    """
    Utility class for training and evaluating classical machine learning models
    (Logistic Regression, Random Forest, XGBoost, Ensemble) on a variety of
    feature sets, including TF-IDF, BERT-based cosine similarity, and their combinations.

    Tracks all fitted models and evaluation results for convenient comparison.
    """

    def __init__(self, n_jobs=-1):
        """
        Initialize the trainer: create empty containers for models, vectorizers, and results.
        Args:
            n_jobs (int): Number of parallel jobs to run for RF/XGB/ensemble (-1 means 'all cores').
        """
        self.models = {}
        self.vectorizers = {}
        self.results = []  # For storing all evaluation results as dicts
        self.n_jobs = n_jobs  # Number of CPU cores to use (set by user)

    def save_model(self, name, path):
        """
        Save a trained model to disk using joblib.
        
        Args:
            name (str): Key of the model in self.models.
            path (str): Path to save the model.
        """
        joblib.dump(self.models[name], path)

    def load_model(self, name, path):
        """
        Load a model from disk and add to self.models.
        
        Args:
            name (str): Name/key for the loaded model.
            path (str): Path where the model is stored.
        """
        self.models[name] = joblib.load(path)

    def train_logreg_tfidf(self, X_train, y_train):
        """
        Train logistic regression using TF-IDF features.

        Args:
            X_train: Sparse matrix of TF-IDF features.
            y_train: Target array.
        Returns:
            Trained LogisticRegression model.
        """
        model = LogisticRegression(max_iter=1000, class_weight='balanced')
        model.fit(X_train, y_train)
        self.models['logreg_tfidf'] = model
        return model

    def train_rf_tfidf(self, X_train, y_train):
        """
        Train random forest using TF-IDF features.

        Args:
            X_train: Sparse matrix of TF-IDF features.
            y_train: Target array.
        Returns:
            Trained RandomForestClassifier model.
        """
        model = RandomForestClassifier(n_estimators=100, n_jobs=self.n_jobs, random_state=42)
        model.fit(X_train, y_train)
        self.models['rf_tfidf'] = model
        return model

    def train_logreg_cosine(self, X_train, y_train):
        """
        Train logistic regression on a single cosine similarity feature.

        Args:
            X_train: 2D array or DataFrame with cosine similarity.
            y_train: Target array.
        Returns:
            Trained LogisticRegression model.
        """
        model = LogisticRegression(max_iter=1000, class_weight='balanced')
        model.fit(X_train, y_train)
        self.models['logreg_cosine'] = model
        return model

    def train_rf_cosine(self, X_train, y_train):
        """
        Train random forest on a single cosine similarity feature.

        Args:
            X_train: 2D array or DataFrame with cosine similarity.
            y_train: Target array.
        Returns:
            Trained RandomForestClassifier model.
        """
        model = RandomForestClassifier(n_estimators=100, n_jobs=-1, random_state=42)
        model.fit(X_train, y_train)
        self.models['rf_cosine'] = model
        return model

    def train_logreg_combined(self, X_train, y_train):
        """
        Train logistic regression on combined features (e.g., TF-IDF + cosine similarity).

        Args:
            X_train: Combined feature matrix (sparse + dense).
            y_train: Target array.
        Returns:
            Trained LogisticRegression model.
        """
        model = LogisticRegression(max_iter=1000, class_weight='balanced')
        model.fit(X_train, y_train)
        self.models['logreg_combined'] = model
        return model

    def train_rf_combined(self, X_train, y_train):
        """
        Train random forest on combined features (e.g., TF-IDF + cosine similarity).

        Args:
            X_train: Combined feature matrix (sparse + dense).
            y_train: Target array.
        Returns:
            Trained RandomForestClassifier model.
        """
        model = RandomForestClassifier(n_estimators=100, n_jobs=self.n_jobs, random_state=42)
        model.fit(X_train, y_train)
        self.models['rf_combined'] = model
        return model

    def train_xgb_combined(self, X_train, y_train, scale_pos_weight):
        """
        Train XGBoost on combined features (e.g., TF-IDF + cosine similarity).

        Args:
            X_train: Combined feature matrix (sparse + dense).
            y_train: Target array.
            scale_pos_weight (float): Class balancing weight for positive class.
        Returns:
            Trained XGBClassifier model.
        """
        model = XGBClassifier(
            n_estimators=300, max_depth=9, learning_rate=0.05,
            scale_pos_weight=scale_pos_weight, random_state=42, n_jobs=self.n_jobs, subsample=0.8, colsample_bytree=0.8
        )
        model.fit(X_train, y_train)
        self.models['xgb_combined'] = model
        return model

    def train_ensemble(self, estimators, X_train, y_train, voting='soft'):
        """
        Train an ensemble (VotingClassifier) on any feature matrix, using specified models.

        Args:
            estimators (list): List of (name, model) tuples.
            X_train: Training feature matrix.
            y_train: Target array.
            voting (str): 'soft' for average probabilities, 'hard' for majority class.
        Returns:
            Trained VotingClassifier model.
        """
        ensemble = VotingClassifier(estimators=estimators, voting=voting, n_jobs=self.n_jobs)
        ensemble.fit(X_train, y_train)
        self.models['ensemble'] = ensemble
        return ensemble

    def evaluate(self, model, X_test, y_test, model_name, feature_set, print_report=True, notes=None):
        """
        Evaluate a model: compute F1, log loss, confusion matrix, and print results.
        Store the results in the internal summary table.

        Args:
            model: Fitted model object.
            X_test: Test feature matrix.
            y_test: Test target array.
            model_name (str): Label for the model (e.g., "LogReg").
            feature_set (str): Label for feature set (e.g., "TF-IDF+BERT").
            print_report (bool): Whether to print full classification report.
            notes (str or None): Optional notes for the summary table.

        Returns:
            dict: { 'f1': ..., 'logloss': ... }
        """
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1]
        f1 = f1_score(y_test, y_pred)
        logloss = log_loss(y_test, y_proba)
        if print_report:
            print(f"\n{model_name} ({feature_set})")
            print("F1-score:", f1)
            print("Log loss:", logloss)
            print("Confusion matrix:\n", confusion_matrix(y_test, y_pred))
            print("Classification report:\n", classification_report(y_test, y_pred))
        self.results.append({
            'Model': model_name,
            'Features': feature_set,
            'F1': f1,
            'LogLoss': logloss,
            'Notes': notes if notes else ''
        })
        return {"f1": f1, "logloss": logloss}

    def summary(self):
        """
        Return all model evaluation results as a pandas DataFrame.

        Returns:
            DataFrame: Table with results of all .evaluate() calls.
        """
        return pd.DataFrame(self.results)

    def feature_importance(self, model, vectorizer=None, feature_names_extra=None, top_n=10):
        """
        Print feature importances for linear/logistic or tree-based models.

        Args:
            model: Fitted model.
            vectorizer: Fitted vectorizer, if used (to show feature names).
            feature_names_extra (list): Names of additional non-TFIDF features, if any.
            top_n (int): Number of top features to print.
        """
        # For linear models (LogisticRegression, etc.)
        if hasattr(model, 'coef_'):
            importances = model.coef_[0]
            feature_names = vectorizer.vectorizer.get_feature_names_out().tolist()
            feature_names = feature_names if vectorizer else []
            if feature_names_extra:
                feature_names.extend(feature_names_extra)
            top_idx = np.argsort(np.abs(importances))[::-1][:top_n]
            for idx in top_idx:
                name = feature_names[idx] if idx < len(feature_names) else f'Feature {idx}'
                print(f"{name}: {importances[idx]:.4f}")
        # For tree-based models (Random Forest, XGBoost, etc.)
        elif hasattr(model, 'feature_importances_'):
            importances = model.feature_importances_
            top_idx = np.argsort(importances)[::-1][:top_n]
            for idx in top_idx:
                print(f"Feature {idx}: {importances[idx]:.4f}")
        else:
            print("Model does not provide feature importances.")
