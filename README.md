**🏀 NBA Player Prop Prediction System**

**Automated ML Pipeline for Predicting NBA Player Prop Outcomes**

This project is a full end-to-end automated system for predicting NBA player prop outcomes using machine learning (XGBoost regression), custom feature engineering, and daily ingestion of betting markets. The system runs on **Google Cloud Platform** and delivers automated **daily predictions**, **ongoing model retraining**, and **weekly calibration** to maintain high predictive accuracy.

**📌 Project Overview**

The system ingests NBA player box scores, schedules, and betting markets to generate real-time predictive outputs for player props (Points, Rebounds, Assists, etc.).

The pipeline includes:

- Robust data cleaning, timezone normalization, and schema reconciliation between multiple datasets.
- Daily prediction pipeline using XGBoost.
- Historical model retraining with model version control.
- Weekly performance evaluation and isotonic calibration.
- Cloud-based automation for 24/7 reliability.

**🧩 Repository Structure**

**1\. Utilities.py**

Core shared utility functions used across all scripts. Includes:

- Date/time parsing and timezone correction
- Fixes for inconsistent EST/UTC formats
- Player/schedule data cleaning and preprocessing
- Feature engineering for model inputs
- Merge helpers to align player stats, schedules, and outcomes
- Logging, API helpers, and general utilities

**2\. runDailyNBAOdds.py**

Daily prediction pipeline.

This script:

- Pulls the latest NBA player props
- Loads the current top-performing models
- Generates predictions for each available prop
- Saves results to cloud storage
- Sends daily alerts
- Uses the most recent, correctly-aligned stat lines

**3\. runDailyModelsUpdate.py**

Daily retraining pipeline.

This script:

- Collects previous day's completed games
- Merges predictions with actual outcomes
- Trains updated XGBoost regression models
- Saves each model with version control
- Logs evaluation metrics for tracking performance

**4\. weeklyModelCalibration.py**

Weekly model evaluation and calibration.

This script:

- Evaluates all recent model versions
- Runs isotonic regression calibration where applicable
- Selects the highest-performing model for production
- Archives older models for reproducibility

**⚙️ ML Architecture**

**Model Type:**

- XGBoost Regressor (Gradient Boosted Trees)

**Key Features Engineered:**

- Rolling player performance stats
- Usage rates and minutes trends
- Opponent defensive strength
- Team pace & matchup context
- Market-based betting line adjustments

**Model Management:**

- Every model version is stored independently
- Weekly calibration may select between:
  - Raw XGBoost model
  - Isotonic-calibrated model

**☁️ Cloud Infrastructure (GCP)**

The entire system runs on **Google Cloud Platform**, including:

- **Compute Engine VM** running scheduled pipelines
- **Cloud Storage Buckets** for:
  - Raw data
  - Cleaned data
  - Model artifacts
- **Cron scheduling** for:
  - Daily predictions
  - Daily retraining
  - Weekly calibration

This ensures high reliability and reproducible experiments.

**📈 Data Sources**

- Kaggle NBA historical player stats
- Kaggle NBA schedule data
- Odds API sportsbook prop lines
- Custom-cleaned matchup + schedule alignment

Advanced preprocessing resolves:

- Inconsistent timezone formats
- EST vs UTC mismatches
- Naive vs offset datetime strings
- Missing or incorrect schedule entries
- Game ID mismatches across datasets

**🔧 Known Issues & Fixes**

**Problem: Inconsistent Date/Timezone Formats**

The dataset provides multiple formats:

- Some timestamps include offsets (2025-11-03 20:00:00-05:00)
- Others are naive (2025-12-01 17:00:00)
- Schedules use ISO8601 (2025-12-01T22:00:00Z)

**Fix Implemented**

- Custom parse_game_date() handles all formats
- Schedule file overrides used for current seasons
- All times normalized to:

| **Column** | **Meaning** |
| --- | --- |
| gameDate | Naive EST version (baseline) |
| gameDateLocal | TZ-aware EST |
| gameDateUTC | UTC (used for merging props) |

This ensures correct alignment between stats, schedules, and props.

**📊 Outputs**

The system generates:

- Daily prediction files
- Feature snapshots for debugging
- Model evaluation logs
- Weekly calibrated model reports
- Archived XGBoost model artifacts

These outputs support analysis, monitoring, and future retraining.
