"""
Model Training & Evaluation Module — FIFA World Cup 2026 Prediction

Trains and evaluates:
- Model A: XGBoost multi-class classifier (home_win / draw / away_win)
- Model B: XGBoost regressor for goal prediction (Poisson-like)
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.metrics import (
    accuracy_score, f1_score, classification_report,
    confusion_matrix, log_loss, mean_absolute_error
)
from sklearn.preprocessing import LabelEncoder
import xgboost as xgb
import joblib
from pathlib import Path
import json


MODELS_DIR = Path(__file__).parent.parent / "models"


def train_outcome_model(X, y, test_size=0.2, random_state=42, verbose=True):
    """
    Train XGBoost classifier for match outcome prediction.
    
    Args:
        X: Feature DataFrame
        y: Target array (0=home_win, 1=draw, 2=away_win)
        test_size: Fraction for test set
        random_state: Random seed
        verbose: Print training progress
    
    Returns:
        model, X_test, y_test, metrics_dict
    """
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    
    if verbose:
        print(f"📊 Training set: {len(X_train)} samples")
        print(f"📊 Test set: {len(X_test)} samples")
        print(f"📊 Features: {X_train.shape[1]}")
        print(f"📊 Class distribution (train): {np.bincount(y_train)}")
    
    # XGBoost parameters optimized for football prediction
    params = {
        "objective": "multi:softprob",
        "num_class": 3,
        "max_depth": 6,
        "learning_rate": 0.1,
        "n_estimators": 200,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "min_child_weight": 5,
        "reg_alpha": 0.1,
        "reg_lambda": 1.0,
        "random_state": random_state,
        "eval_metric": "mlogloss",
        "use_label_encoder": False,
    }
    
    model = xgb.XGBClassifier(**params)
    
    # Train with early stopping
    model.fit(
        X_train, y_train,
        eval_set=[(X_test, y_test)],
        verbose=verbose,
    )
    
    # Evaluate
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)
    
    metrics = {
        "accuracy": round(accuracy_score(y_test, y_pred), 4),
        "f1_macro": round(f1_score(y_test, y_pred, average="macro"), 4),
        "f1_weighted": round(f1_score(y_test, y_pred, average="weighted"), 4),
        "log_loss": round(log_loss(y_test, y_proba), 4),
        "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
        "classification_report": classification_report(y_test, y_pred, 
            target_names=["Home Win", "Draw", "Away Win"], output_dict=True),
    }
    
    if verbose:
        print(f"\n{'='*50}")
        print(f"🎯 Model Performance:")
        print(f"   Accuracy: {metrics['accuracy']:.1%}")
        print(f"   F1 (macro): {metrics['f1_macro']:.4f}")
        print(f"   Log Loss: {metrics['log_loss']:.4f}")
        print(f"\n{classification_report(y_test, y_pred, target_names=['Home Win', 'Draw', 'Away Win'])}")
    
    return model, X_test, y_test, metrics


def train_goals_model(X, home_goals, away_goals, test_size=0.2, random_state=42, verbose=True):
    """
    Train XGBoost regressors for predicting goals scored by each team.
    
    Returns:
        home_model, away_model, metrics
    """
    X_train, X_test, hg_train, hg_test, ag_train, ag_test = train_test_split(
        X, home_goals, away_goals, test_size=test_size, random_state=random_state
    )
    
    params = {
        "objective": "count:poisson",
        "max_depth": 5,
        "learning_rate": 0.1,
        "n_estimators": 150,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "random_state": random_state,
    }
    
    if verbose:
        print("⚽ Training home goals model...")
    home_model = xgb.XGBRegressor(**params)
    home_model.fit(X_train, hg_train, eval_set=[(X_test, hg_test)], verbose=False)
    
    if verbose:
        print("⚽ Training away goals model...")
    away_model = xgb.XGBRegressor(**params)
    away_model.fit(X_train, ag_train, eval_set=[(X_test, ag_test)], verbose=False)
    
    # Evaluate
    hg_pred = home_model.predict(X_test)
    ag_pred = away_model.predict(X_test)
    
    metrics = {
        "home_goals_mae": round(mean_absolute_error(hg_test, hg_pred), 4),
        "away_goals_mae": round(mean_absolute_error(ag_test, ag_pred), 4),
        "home_goals_avg_pred": round(hg_pred.mean(), 2),
        "away_goals_avg_pred": round(ag_pred.mean(), 2),
    }
    
    if verbose:
        print(f"\n⚽ Goals Model Performance:")
        print(f"   Home Goals MAE: {metrics['home_goals_mae']:.4f}")
        print(f"   Away Goals MAE: {metrics['away_goals_mae']:.4f}")
    
    return home_model, away_model, metrics


def cross_validate_model(X, y, n_folds=5, verbose=True):
    """Run stratified K-fold cross-validation."""
    model = xgb.XGBClassifier(
        objective="multi:softprob",
        num_class=3,
        max_depth=6,
        learning_rate=0.1,
        n_estimators=200,
        use_label_encoder=False,
        eval_metric="mlogloss",
    )
    
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
    scores = cross_val_score(model, X, y, cv=skf, scoring="accuracy")
    
    if verbose:
        print(f"\n📊 {n_folds}-Fold Cross-Validation:")
        print(f"   Scores: {[round(s, 4) for s in scores]}")
        print(f"   Mean: {scores.mean():.4f} ± {scores.std():.4f}")
    
    return scores


def get_feature_importance(model, feature_names=None, top_n=20):
    """
    Get feature importance from trained model.
    
    Returns sorted list of (feature_name, importance) tuples.
    """
    importances = model.feature_importances_
    
    if feature_names is None:
        feature_names = [f"feature_{i}" for i in range(len(importances))]
    
    feat_imp = list(zip(feature_names, importances))
    feat_imp.sort(key=lambda x: x[1], reverse=True)
    
    return feat_imp[:top_n]


def save_model(model, name="xgb_match_predictor", metadata=None):
    """Save model to disk with metadata."""
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    
    model_path = MODELS_DIR / f"{name}.pkl"
    joblib.dump(model, model_path)
    
    if metadata:
        meta_path = MODELS_DIR / f"{name}_metadata.json"
        with open(meta_path, "w") as f:
            # Convert numpy types for JSON serialization
            clean_meta = {}
            for k, v in metadata.items():
                if isinstance(v, np.floating):
                    clean_meta[k] = float(v)
                elif isinstance(v, np.integer):
                    clean_meta[k] = int(v)
                elif isinstance(v, np.ndarray):
                    clean_meta[k] = v.tolist()
                else:
                    clean_meta[k] = v
            json.dump(clean_meta, f, indent=2)
    
    print(f"💾 Model saved: {model_path}")
    return model_path


def load_model(name="xgb_match_predictor"):
    """Load model from disk."""
    model_path = MODELS_DIR / f"{name}.pkl"
    
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")
    
    model = joblib.load(model_path)
    
    # Load metadata if exists
    meta_path = MODELS_DIR / f"{name}_metadata.json"
    metadata = None
    if meta_path.exists():
        with open(meta_path, "r") as f:
            metadata = json.load(f)
    
    return model, metadata
