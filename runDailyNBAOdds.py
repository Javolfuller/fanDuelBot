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
	collect_and_merge_predictions_and_outcomes, check_sports_schedule, get_active_model_for_production)

# Keep readible formatting for probability
pd.set_option('display.float_format', '{:.2f}'.format)

# Load config file
config = configparser.ConfigParser()
config_path = os.path.join(os.path.dirname(__file__), "config.ini")
config.read(os.path.abspath(config_path))

# Get Creds
KAGGLE_USERNAME = config["KAGGLE"]["KAGGLE_USERNAME"]
KAGGLE_KEY = config["KAGGLE"]["KAGGLE_API_KEY"]
ODDS_500_KEY = config["ODDS"]["ODDS_500_KEY"]
ODDS_20K_KEY = config["ODDS"]["ODDS_20K_KEY"]
ODDS_5M_KEY = config["ODDS"]["ODDS_5M_KEY"]

# Establish Sport
SPORT = 'basketball_nba'

# Set variable for kaggle
os.environ["KAGGLE_USERNAME"] = KAGGLE_USERNAME
os.environ["KAGGLE_KEY"] = KAGGLE_KEY


####### Step 1 - Data Collection #######
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


no_games_scheduled = check_sports_schedule(SPORT)
if no_games_scheduled:
	print('No games scheduled for today')
	############# Send Email #############
	exit()

# Collect Upcoming Events
events_df = get_current_events(ODDS_20K_KEY, SPORT)
if events_df.empty:
	print("No Current Events found for today specifically")
	exit()

player_props_df = get_current_player_props(events_df, SPORT, ODDS_20K_KEY)
if player_props_df.empty:
	print("No upcoming player props found from raw API requests for the event list given")
	exit()

props_boxscore_df = player_props_df.merge(refined_player_stats_df, left_on=['fullNameId'], right_on=['fullNameId'])
if props_boxscore_df.empty:
	print("No refined player stats maps to any current player props")
	exit()

props_boxscore_df = props_boxscore_df[props_boxscore_df['eventDateTime'].dt.tz_convert("US/Eastern").dt.date == datetime.now(ZoneInfo("America/New_York")).date()]													
if props_boxscore_df.empty:
	print("No upcoming Games for today")
	exit()

# Adjust "home" field to ensure that upcoming game is either home or away 
props_boxscore_df['home'] = props_boxscore_df.apply(lambda row: 1 if row['homeTeam'] == row['playerteamName'] else 0, axis=1)

####### Step 2 - Data Cleaning/ Prepping for ML #######
props_boxscore_df = clean_props_with_boxscore(props_boxscore_df, 'numMinutes', 'gameType')
props_boxscore_df = props_boxscore_df.sort_values('gameDate', ascending=False).drop_duplicates(['eventId', 'personId', 'market', 'overUnder'])
props_boxscore_market_dict = market_splitter(props_boxscore_df)



####### Step 3 - Stat Feature Selection, make predictions, and Compute Probabilities for each Available Market #######

props_box_predicts = []
for market in props_boxscore_market_dict.keys():

	print(f'Evaluation Predictions for {market}.....')

	# Contain next game stat line based on market to avoid feature leakage
	next_game_stats = ['nextGamePoints', 'nextGameAssists', 'nextGameThreePointersMade', 'nextGameReboundsTotal', 'nextGameSteals', 'nextGameBlocks']
	stat_type = market.split('_')[1]
	props_box_stat = props_boxscore_market_dict[market]

	if stat_type == 'threes':
		drop_columns = [col for col in next_game_stats if "three" in col.lower()]
	else:
		drop_columns = [col for col in next_game_stats if stat_type in col.lower()]

	props_box_stat = props_box_stat.drop(drop_columns, axis=1, errors='ignore')
	print(f"Total Merged {stat_type} Props", len(props_box_stat))

	# Get Active production model 
	ml_features = ml_feature_selection(SPORT, market)
	active_model, active_model_path = get_active_model_for_production(sport=SPORT, stat_type=stat_type)

	# Get Predictions
	if 'isotonic' in active_model_path.lower():
		calibrator = active_model['calibrator']
		active_model = active_model['xgb_progressive_model']
		active_model.set_params(n_jobs=1)

		raw_predictions = active_model.predict(props_box_stat[ml_features])
		predictions = calibrator.predict(raw_predictions)
	else:
		predictions = active_model.predict(props_box_stat[ml_features])

	del active_model
	gc.collect()

	# Output Columns
	props_box_stat[f'predicted{stat_type.title()}'] = np.round(predictions).astype(int)
	props_box_stat['predictedLineOutCome'] = props_box_stat.apply(lambda row: predict_outcome_label(row, f'predicted{stat_type.title()}', row['line']), axis=1)
	props_box_stat['probability'] = (1 - props_box_stat.apply(lambda row: norm.cdf(row['line'], loc=row[f'predicted{stat_type.title()}'], scale=3.3389), axis=1 )) * 100
	props_box_stat['modelUsed'] = active_model_path.split('/')[-1]
	props_box_predicts.append(props_box_stat)
	print()

props_box_predicts = pd.concat(props_box_predicts, ignore_index=True) # Combine Predictions

####### Step 4 - Save Predictions #######
file_path = os.path.join( "outcomes", "daily", "predictions", SPORT,
            f"daily_predictions_{SPORT}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.parquet")
os.makedirs(os.path.dirname(file_path), exist_ok=True)
props_box_predicts.to_parquet(file_path, engine="pyarrow", index=False)

####### Step 5 - Send email #######
daily_predictions_df = props_box_predicts.sort_values('probability', ascending=False).drop_duplicates(['eventId', 'personId', 'market'])
email_betting_odds(daily_predictions_df)




