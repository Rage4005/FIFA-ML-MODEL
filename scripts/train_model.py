"""
Model Training Script — FIFA World Cup 2026 Prediction

End-to-end pipeline:
1. Generate/load data
2. Build features
3. Train models
4. Evaluate & save

Usage:
    python scripts/train_model.py
"""

import sys
import os
import time

# Add project root to path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from src import data_loader, features, model as model_module


def main():
    print("=" * 60)
    print("🏆 FIFA World Cup 2026 — Model Training Pipeline")
    print("=" * 60)
    
    start_time = time.time()
    
    # ----- Step 1: Generate Data (if not exists) -----
    raw_dir = os.path.join(PROJECT_ROOT, "data", "raw")
    if not os.path.exists(os.path.join(raw_dir, "international_matches.csv")):
        print("\n📦 Step 1: Generating training data...")
        from scripts.generate_data import generate_all_data
        data_dir = os.path.join(PROJECT_ROOT, "data")
        generate_all_data(data_dir)
    else:
        print("\n📦 Step 1: Data already exists, loading...")
    
    # ----- Step 2: Load Data -----
    print("\n📂 Step 2: Loading datasets...")
    matches_df = data_loader.load_matches()
    rankings_df = data_loader.load_rankings()
    elo_df = data_loader.load_elo_ratings()
    metadata_df = data_loader.load_team_metadata()
    
    print(f"   📊 {len(matches_df)} matches loaded")
    print(f"   🏅 {len(rankings_df)} ranking records")
    print(f"   📈 {len(elo_df)} Elo records")
    print(f"   👥 {len(metadata_df)} teams with metadata")
    
    # ----- Step 3: Feature Engineering -----
    print("\n🔧 Step 3: Building features...")
    print("   (This may take a few minutes for large datasets...)")
    
    # Use a sample for faster training
    X, y = features.build_training_dataset(
        matches_df, rankings_df, elo_df, metadata_df,
        start_year=2015,
        sample_size=3000,
        verbose=True
    )
    
    print(f"\n   📊 Training dataset shape: {X.shape}")
    print(f"   📊 Feature names: {list(X.columns[:10])}... ({len(X.columns)} total)")
    print(f"   📊 Class distribution: Home Win={sum(y==0)}, Draw={sum(y==1)}, Away Win={sum(y==2)}")
    
    # Save processed data
    processed_dir = os.path.join(PROJECT_ROOT, "data", "processed")
    os.makedirs(processed_dir, exist_ok=True)
    X.to_csv(os.path.join(processed_dir, "training_features.csv"), index=False)
    
    import numpy as np
    np.save(os.path.join(processed_dir, "training_labels.npy"), y)
    print("   💾 Training data saved to data/processed/")
    
    # ----- Step 4: Train Outcome Model -----
    print("\n🤖 Step 4: Training match outcome model (XGBoost)...")
    outcome_model, X_test, y_test, metrics = model_module.train_outcome_model(
        X, y, test_size=0.2, verbose=True
    )
    
    # ----- Step 5: Cross-Validation -----
    print("\n📊 Step 5: Cross-validation...")
    cv_scores = model_module.cross_validate_model(X, y, n_folds=5, verbose=True)
    
    # ----- Step 6: Feature Importance -----
    print("\n🔍 Step 6: Feature importance analysis...")
    feat_imp = model_module.get_feature_importance(outcome_model, list(X.columns), top_n=15)
    print("\n   Top 15 Features:")
    for i, (name, imp) in enumerate(feat_imp, 1):
        bar = "█" * int(imp * 200)
        print(f"   {i:2d}. {name:<40} {imp:.4f} {bar}")
    
    # ----- Step 7: Save Model -----
    print("\n💾 Step 7: Saving model...")
    model_metadata = {
        "accuracy": metrics["accuracy"],
        "f1_macro": metrics["f1_macro"],
        "log_loss": metrics["log_loss"],
        "cv_mean": float(cv_scores.mean()),
        "cv_std": float(cv_scores.std()),
        "n_features": X.shape[1],
        "n_training_samples": X.shape[0],
        "feature_names": list(X.columns),
        "top_features": [(n, float(v)) for n, v in feat_imp[:10]],
    }
    
    model_module.save_model(outcome_model, "xgb_match_predictor", model_metadata)
    
    # ----- Step 8: Quick Test -----
    print("\n🧪 Step 8: Quick prediction test...")
    from src.predict import predict_match
    
    test_matches = [
        ("Brazil", "Spain"),
        ("Argentina", "France"),
        ("England", "Germany"),
    ]
    
    for team_a, team_b in test_matches:
        try:
            result = predict_match(team_a, team_b, model=outcome_model, verbose=True)
        except Exception as e:
            print(f"   ⚠️  {team_a} vs {team_b}: {e}")
    
    elapsed = time.time() - start_time
    print(f"\n{'='*60}")
    print(f"✅ Training complete in {elapsed:.1f} seconds!")
    print(f"{'='*60}")
    print(f"\nNext steps:")
    print(f"  1. Run the Streamlit dashboard: streamlit run dashboard/app.py")
    print(f"  2. Check model insights in the dashboard")
    print(f"  3. Run tournament simulation from the dashboard")


if __name__ == "__main__":
    main()
