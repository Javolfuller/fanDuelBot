import os 
# Cap computation to on thread
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
import gc
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')
from bettingBotUtil.Utilities import (fetch_nba_data_sets, refine_nba_stats, build_features, collect_all_player_props, clean_props_with_boxscore,
   get_latest_model_in_log, prop_model_evaluation, predict_outcome_label, evaluate_predictions, replace_active_models)

# Establish Sport
SPORT = 'basketball_nba'

###### STEP 1 - Collect Data ######

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

all_props = collect_all_player_props()
refined_player_stats_df = refined_player_stats_df[refined_player_stats_df['nextGameDateUTC'].notna()].copy()

props_and_box = pd.merge_asof(all_props.sort_values('eventDateTime'), refined_player_stats_df.sort_values('nextGameDateUTC'),
								left_on='eventDateTime', right_on='nextGameDateUTC', by=['fullNameId'],
								tolerance=pd.Timedelta('2h'), direction='nearest')
props_and_box['home'] = props_and_box.apply(lambda row: 1 if row['homeTeam'] == row['playerteamName'] else 0, axis=1)
props_and_box = clean_props_with_boxscore(props_and_box, 'numMinutes', 'gameType')

####### STEP 2 - Assess Each Recent Model Type #######
all_results = []

for mod_type in ['progressive', 'progressive_isotonic']:
    for stat in ['points', 'assists', 'threes', 'rebounds', 'steals', 'blocks']:
        print(f"Evaluating {stat} with {mod_type} model...")

        # Get latest per model type/stat
        market_col, target_col, latest_model, latest_mod_path = get_latest_model_in_log(sport=SPORT, stat_type=stat, model_type=mod_type)

        predicts = prop_model_evaluation(model=latest_model, model_type=mod_type, model_path=latest_mod_path, 
                                         stat_type=stat, props_box_df=props_and_box, sport=SPORT)
        if not predicts.empty:
            all_results.append(predicts)
all_predictions_df = pd.concat(all_results, ignore_index=True)

# Collapse outcome stats to one column, drop columns
all_predictions_df['outcomeStat'] = all_predictions_df.apply(lambda row: (
                                    row['nextGamePoints'] if row['market'] == 'player_points' else
                                    row['nextGameAssists'] if row['market'] == 'player_assists' else
                                    row['nextGameThreePointersMade'] if row['market'] == 'player_threes' else
                                    row['nextGameReboundsTotal'] if row['market'] == 'player_rebounds' else
                                    row['nextGameSteals'] if row['market'] == 'player_steals' else
                                    row['nextGameBlocks']), axis=1)

all_predictions_df = all_predictions_df.drop(['nextGamePoints', 'nextGameAssists', 'nextGameThreePointersMade', 'nextGameReboundsTotal', 'nextGameSteals', 'nextGameBlocks'], axis=1)
all_predictions_df['predictedLineOutCome'] = all_predictions_df.apply(lambda row: predict_outcome_label(row, 'modelPrediction', row['line']), axis=1)
all_predictions_df = evaluate_predictions(all_predictions_df)

######## STEP 3 - Compute Group Bys ########

# RMSE
rmse_df = (
    all_predictions_df
        .groupby(['modelType', 'marketType'])
        .apply(lambda g: np.sqrt(np.mean((g['modelPrediction'] - g['outcomeStat'])**2)))
        .reset_index(name='RMSE')
)

# Correct Predictions
correct_count_df = (
    all_predictions_df
        .groupby(['modelType', 'marketType'])['predictionCorrect']
        .sum()
        .reset_index(name='numCorrect')
)

sample_count_df = (
    all_predictions_df
        .groupby(['modelType', 'marketType'])['predictionCorrect']
        .count()
        .reset_index(name='numSamples')
)

# Combine
evaluation_df = (
    rmse_df
    .merge(correct_count_df, on=['modelType', 'marketType'])
    .merge(sample_count_df, on=['modelType', 'marketType'])
)

evaluation_df['accuracy'] = evaluation_df['numCorrect'] / evaluation_df['numSamples']

best_models = (evaluation_df
        .sort_values(['RMSE'], ascending=[True])
        .groupby('marketType')
        .head(1))

######## STEP 4 - Select and Update Production Model ########

best_models['model_filter'] = best_models['modelType']+ '-' + best_models['marketType'] # create a collumn to filter on
all_predictions_df['model_filter'] = all_predictions_df['modelType']+ '-' + all_predictions_df['marketType']
path_to_best_models = all_predictions_df[all_predictions_df.model_filter.isin(best_models.model_filter)]['modelPath'].unique()
replace_active_models(model_path_list=path_to_best_models, sport=SPORT, stat_list=['points', 'assists', 'threes', 'rebounds', 'steals', 'blocks']) # update production models




