# Can We Predict Post-Transfer Success? A Machine Learning Approach using Player and Team Data

## 📋 Project Overview

This project aims to predict a football player's performance in their new team using pre-transfer data. By leveraging machine learning techniques, we develop models that can forecast post-transfer success, providing valuable insights for scouting and risk analysis in football transfers.

## 🎯 Research Objective

**Main Question:** Can we predict whether a player will be successful in their new team based on their pre-transfer characteristics and historical performance data?

**Practical Value:** This research provides data-driven support for:
- Transfer decision-making processes
- Scouting and talent identification
- Risk assessment in player acquisitions
- Strategic planning for football clubs

## 📊 Success Metrics

We measure transfer success at two levels:

### A. Individual Performance-Based Success

Player-specific measurable indicators:

| Metric | Description | Purpose |
|--------|-------------|---------|
| **Minutes Played** | Total playing time | Indicator of coach's trust |
| **Goals + Assists** | Direct attacking contribution | Offensive impact measurement |
| **Expected Goals (xG) / Expected Assists (xA)** | Quality of chances created/taken | Performance quality assessment |
| **Player Rating** | Average match rating (Sofascore, WhoScored, FBref) | Overall match performance |
| **Injury Days Missed** | Days unavailable due to injury | Availability and reliability factor |
| **Performance Change (%)** | Pre-transfer vs post-transfer comparison | Direct impact measurement |

### B. Team Contribution-Based Success

Transfer's economic or sporting contribution to the team:

- **Team Points Average Change:** Impact on team's point-per-game ratio
- **Trophy Wins / Ranking Improvement:** Tangible success indicators
- **Team's Offensive/Defensive Metrics:** Improvement in team statistics

## 🤖 Machine Learning Approach

### Model Type
**Regression and Classification Models**

We employ both approaches:
1. **Regression:** Predicting continuous performance metrics (e.g., "goals per 90 minutes after transfer")
2. **Classification:** Categorizing transfers as "successful" or "unsuccessful"

### Variables

**Dependent Variable (Target):**
- Performance metrics after transfer (e.g., goals per 90 min, assists, rating)
- Binary success classification (successful/unsuccessful transfer)

**Independent Variables (Features):**
- Player age
- Playing position
- Historical performance statistics
- Market value
- League level (origin and destination)
- Previous club performance
- Transfer fee
- Contract length
- Physical attributes

### Modeling Strategy

```
Pre-Transfer Data → Machine Learning Model → Post-Transfer Performance Prediction
```

**Goal:** Predict whether a transfer will be successful before it happens, enabling proactive decision-making.

## 🔧 Technical Stack

- **Programming Language:** Python 3.11+
- **Data Collection:** Web scraping (BeautifulSoup, Selenium), APIs
- **Data Processing:** Pandas, NumPy
- **Machine Learning:** Scikit-learn, XGBoost, LightGBM
- **Visualization:** Matplotlib, Seaborn, Plotly
- **Statistical Analysis:** SciPy, Statsmodels

## 📁 Project Structure

```
transfer-success-prediction/
├── data/
│   ├── raw/                 # Raw scraped data
│   ├── processed/           # Cleaned and processed datasets
│   └── features/            # Engineered features
├── notebooks/
│   ├── 01_data_collection.ipynb
│   ├── 02_data_cleaning.ipynb
│   ├── 03_exploratory_analysis.ipynb
│   ├── 04_feature_engineering.ipynb
│   └── 05_modeling.ipynb
├── src/
│   ├── data_collection/     # Scraping scripts
│   ├── preprocessing/       # Data cleaning functions
│   ├── features/            # Feature engineering
│   └── models/              # Model training and evaluation
├── results/
│   ├── figures/             # Visualizations
│   └── models/              # Saved models
├── requirements.txt
└── README.md
```

## 🚀 Getting Started

### Prerequisites

```bash
python >= 3.11
pip install -r requirements.txt
```

### Installation

```bash
git clone https://github.com/mehmetyalc/transfer-success-prediction.git
cd transfer-success-prediction
pip install -r requirements.txt
```

### Usage

1. **Data Collection:** Run data scraping scripts in `src/data_collection/`
2. **Data Processing:** Execute notebooks in order (01 → 05)
3. **Model Training:** Use `src/models/train.py` to train models
4. **Prediction:** Use trained models to predict transfer success

## 📈 Expected Outcomes

- **Predictive Models:** Trained ML models capable of forecasting post-transfer performance
- **Feature Importance Analysis:** Identification of key factors influencing transfer success
- **Performance Benchmarks:** Model accuracy, precision, recall, and F1-scores
- **Actionable Insights:** Data-driven recommendations for transfer decisions

## 📚 Data Sources

- [Transfermarkt](https://www.transfermarkt.com/) - Transfer fees and market values
- [FBref](https://fbref.com/) - Advanced football statistics
- [WhoScored](https://www.whoscored.com/) - Player ratings and match data
- [Sofascore](https://www.sofascore.com/) - Performance ratings

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 📧 Contact

For questions or collaboration opportunities, please open an issue or contact the repository owner.

## 🔗 Related Projects

- [Football Transfer Economic Efficiency Analysis](https://github.com/mehmetyalc/transfer-economic-efficiency)

---

**Note:** This is an academic research project. Results should be used as supplementary information alongside expert scouting and domain knowledge.

