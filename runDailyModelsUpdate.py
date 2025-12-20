import os 
import gc
# Cap computation to on thread
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
import pandas as pd
from datetime import datetime, timezone
import warnings
warnings.filterwarnings("ignore", category=pd.errors.PerformanceWarning)
from bettingBotUtil.Utilities import (fetch_nba_data_sets, refine_nba_stats, build_features, get_latest_model_in_log, collect_all_player_props,
update_xgb_progressive_models)
import psutil

sport_kind = 'basketball_nba'
stat_list = ['points', 'assists', 'threes', 'rebounds', 'steals', 'blocks']
model_kind = 'progressive' # no need to pass in Isotonic types, if base progressive is updated a new isotonic is made


for stat_kind in stat_list:
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
    
    market_col, target_col, active_xgb_model, active_xgb_path = get_latest_model_in_log(sport=sport_kind, stat_type=stat_kind, model_type=model_kind)
    props = collect_all_player_props()
    active_xgb_model.set_params(n_jobs=1)
    update_xgb_progressive_models(stats_df=refined_player_stats_df, props_df=props, market=market_col, stat_type=stat_kind, 
                                  model_type=model_kind, target=target_col, sport=sport_kind, current_xgb_model=active_xgb_model, 
                                  og_model_path=active_xgb_path)
    del active_xgb_model
    gc.collect()
