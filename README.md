# ⚽ FIFA World Cup 2026 AI Predictor

An AI-powered simulation and prediction engine for the FIFA World Cup 2026. This project uses machine learning (XGBoost) and Monte Carlo simulations to model match outcomes, predict scores, and run tournament brackets from the group stage to the final.

---

## 🚀 How to Run

### 1. Install Dependencies
Ensure you have Python installed, then install the required libraries:
```bash
pip install -r requirements.txt
```

### 2. Generate Data & Train the Model
The repository does not contain raw or processed data files (they are excluded via `.gitignore`). To bootstrap the project, run the training script. If no local datasets exist, it will automatically generate a realistic simulation dataset to get you started immediately:
```bash
python scripts/train_model.py
```

### 3. Launch the Dashboard
Run the Streamlit interactive dashboard to run predictions, tournament brackets, and view model feature importance:
```bash
python -m streamlit run dashboard/app.py
```

---

## 📊 Data Sourcing

For real-world prediction accuracy, you can download real-world datasets and place them under `data/raw/` with schemas matching those defined in `src/data_loader.py`. 

Here are the recommended sources to download the actual datasets:

1. **International Match History:**
   * **Source:** [Kaggle - International Football Results from 1872 to Present](https://www.kaggle.com/datasets/martj42/international-football-results-from-1872-to-2017)
   * **Filename in project:** `data/raw/international_matches.csv`

2. **FIFA Men's World Rankings:**
   * **Source:** [Kaggle - FIFA Men's World Rankings](https://www.kaggle.com/datasets/cashnarry/fifa-world-rankings-19922021) or official rankings from FIFA's website.
   * **Filename in project:** `data/raw/fifa_rankings.csv`

3. **Elo Ratings:**
   * **Source:** National team ratings from [World Football Elo Ratings (eloratings.net)](https://www.eloratings.net/).
   * **Filename in project:** `data/raw/elo_ratings.csv`

4. **Squad Attributes & Market Values:**
   * **Source:** Squad values, average age, and league player counts from [Transfermarkt](https://www.transfermarkt.com/).
   * **Filename in project:** `data/raw/team_metadata.csv`

---

## 🛠️ Project Structure

* `dashboard/` — Streamlit pages and visual components
* `scripts/` — Scripts for generating data (`generate_data.py`) and model training (`train_model.py`)
* `src/` — Core modules for feature engineering, data loading, predictions, and Monte Carlo tournament simulation
* `models/` — Model metadata and feature importances
