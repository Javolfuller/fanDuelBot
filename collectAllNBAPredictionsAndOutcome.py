import os
# Cap computation to on thread
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
import gc
import configparser
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings("ignore")
from datetime import datetime, timedelta, time, timezone
from zoneinfo import ZoneInfo
from dateutil.relativedelta import relativedelta
import joblib
import requests
from time import sleep
from kaggle.api.kaggle_api_extended import KaggleApi
from xgboost import XGBRegressor
from scipy.stats import norm
from bettingBotUtil.Utilities import (fetch_nba_data_sets, refine_nba_stats, build_features, predict_outcome_label, outcome_label, get_current_events,
	get_current_player_props, clean_props_with_boxscore, market_splitter, ml_feature_selection, predict_outcome_label, email_betting_odds,
	collect_and_merge_predictions_and_outcomes, check_sports_schedule, get_active_model_for_production, parse_game_date)

# Fetch nba stats
player_stats, team_stats, _  = fetch_nba_data_sets()

# Clean player box score stats, get latest
refined_player_stats_df = refine_nba_stats(player_stats, team_stats)
refined_player_stats_df = build_features(refined_player_stats_df,
   ['points', 'assists', 'blocks', 'steals', 'threePointersMade', 'threePointersAttempted', 'reboundsOffensive', 'reboundsTotal',
    'plusMinusPoints', 'usageRate','trueShooting', 'effectiveFieldGoal', 'freeThrowsPercentage', 'fieldGoalsPercentage',
    'threePointersPercentage', 'totalReboundingRate', 'opponentTeamBlocks', 'opponentTeamSteals', 'opponentTeamPlusMinusPoints',
    'opponentTeamTotalReboundingRate', 'opponentDefensiveRating', 'numMinutes', 'turnovers',
    'opponentTeamReboundsOffensive', 'opponentTeamReboundsDefensive', 'opponentTeamFieldGoalsAttempted',
    'opponentTeamFieldGoalsMade', 'opponentTeamFieldGoalsPercentage'], 'gameDate')

#####################################
############# HOT FIX ###############
#####################################

schedule = pd.read_csv("nba_data/LeagueSchedule25_26.csv")
player_stats = player_stats.merge(schedule[['gameId', 'gameDateTimeEst']], on='gameId', how='left')
player_stats['gameDateTimeEst'] = player_stats.apply(lambda x: x['gameDateTimeEst_x'] if pd.isna(x['gameDateTimeEst_y']) else x['gameDateTimeEst_y'], axis=1)
player_stats = player_stats.drop(['gameDateTimeEst_x', 'gameDateTimeEst_y'], axis=1)

#####################################
############# HOT FIX ###############
#####################################

# Convert date time columns
player_stats['gameDate'] = player_stats['gameDateTimeEst'].apply(parse_game_date)  # stays naive at first
player_stats['gameDateLocal'] = player_stats['gameDate'].dt.tz_localize("US/Eastern")
# Also convert to UTC for alignment with Odds API props
player_stats['gameDateUTC'] = player_stats['gameDateLocal'].dt.tz_convert("UTC")
# Breakouts for merging flexibility
player_stats['gameDateLocal_date'] = player_stats['gameDateLocal'].dt.date
player_stats['gameDateLocal_time'] = player_stats['gameDateLocal'].dt.time
player_stats['gameDateLocal_year'] = player_stats['gameDateLocal'].dt.year
player_stats['gameDateUTC_date'] = player_stats['gameDateUTC'].dt.date
player_stats['gameDateUTC_time'] = player_stats['gameDateUTC'].dt.time
player_stats['gameDateUTC_hour'] = player_stats['gameDateUTC'].dt.hour

all_predictions = collect_and_merge_predictions_and_outcomes(player_stats, 'basketball_nba')
all_predictions = all_predictions.drop_duplicates(['eventId', 'personId', 'market'])

all_predictions.to_parquet("./outcomes/results/basketball_nba/basketball_nba_2025.parquet")
