import configparser
import copy
import json
import os
import shutil
import smtplib
import warnings
from datetime import datetime, time, timedelta, timezone
from email.message import EmailMessage
from time import sleep
from zoneinfo import ZoneInfo

import joblib
import numpy as np
import pandas as pd
import pytz
import requests
from dateutil import parser
from dateutil.relativedelta import relativedelta
from kaggle.api.kaggle_api_extended import KaggleApi
from sklearn import metrics
from sklearn.isotonic import IsotonicRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

pd.set_option("display.float_format", "{:.2f}".format)

# Load config file
config = configparser.ConfigParser()
config_path = os.path.join(os.path.dirname(__file__), "..", "config.ini")
config.read(os.path.abspath(config_path))
KAGGLE_USERNAME = config["KAGGLE"]["KAGGLE_USERNAME"]
KAGGLE_KEY = config["KAGGLE"]["KAGGLE_API_KEY"]
EMAIL_ADDRESS = config["EMAIL"]["EMAIL_ADDRESS"]
EMAIL_PWD = config["EMAIL"]["EMAIL_PWD"]

# set variable for kaggle
os.environ["KAGGLE_USERNAME"] = KAGGLE_USERNAME
os.environ["KAGGLE_KEY"] = KAGGLE_KEY


def fetch_nba_data_sets():
    api = KaggleApi()
    api.authenticate()

    # Download and unzip to folder
    api.dataset_download_files(
        "eoinamoore/historical-nba-data-and-player-box-scores",
        path="nba_data",
        unzip=True,
    )

    # read in player
    players = pd.read_csv("nba_data/Players.csv")
    # read teams stats
    team_stats = pd.read_csv("nba_data/TeamStatistics.csv")
    # read boxscores
    player_stats = pd.read_csv("nba_data/PlayerStatistics.csv", low_memory=False)

    # Optimize memory by converting datatypes
    for col in player_stats.select_dtypes(include=["int64"]).columns:
        player_stats[col] = pd.to_numeric(player_stats[col], downcast="integer")

    for col in player_stats.select_dtypes(include=["float64"]).columns:
        player_stats[col] = pd.to_numeric(player_stats[col], downcast="float")

    return player_stats, team_stats, players


def parse_game_date(dt):
    """Handles both old 'YYYY-MM-DD HH:MM:SS' and new 'YYYY-MM-DDTHH:MM:SSZ' formats."""
    if pd.isna(dt):
        return pd.NaT
    dt_str = str(dt)
    try:
        # Try parsing ISO format (with Z)
        parsed = parser.isoparse(dt_str)
    except ValueError:
        # Fallback for old naive format
        parsed = pd.to_datetime(dt_str)
    # Return naive datetime so tz_localize works as before
    return parsed.replace(tzinfo=None) if parsed.tzinfo is not None else parsed


def refine_nba_stats(player_stats, team_stats, cut_off=5):
    """
    Refines player box score stats dataset by cleaning, adjusting datatypes, adding advanced stats and trimming fields for final output
    Requires box score dataframe and team boxscore dataframe
    """

    # 1 -Initial Load and cleaning

    #####################################
    ############# HOT FIX ###############
    #####################################

    schedule = pd.read_csv("nba_data/LeagueSchedule25_26.csv")
    player_stats = player_stats.merge(
        schedule[["gameId", "gameDateTimeEst"]], on="gameId", how="left"
    )
    player_stats["gameDateTimeEst"] = player_stats.apply(
        lambda x: x["gameDateTimeEst_x"]
        if pd.isna(x["gameDateTimeEst_y"])
        else x["gameDateTimeEst_y"],
        axis=1,
    )
    player_stats = player_stats.drop(["gameDateTimeEst_x", "gameDateTimeEst_y"], axis=1)

    #####################################
    ############# HOT FIX ###############
    #####################################

    # Convert date time columns
    player_stats["gameDate"] = player_stats["gameDateTimeEst"].apply(
        parse_game_date
    )  # stays naive at first
    player_stats["gameDateLocal"] = player_stats["gameDate"].dt.tz_localize(
        "US/Eastern"
    )
    # Also convert to UTC for alignment with Odds API props
    player_stats["gameDateUTC"] = player_stats["gameDateLocal"].dt.tz_convert("UTC")
    # Breakouts for merging flexibility
    player_stats["gameDateLocal_date"] = player_stats["gameDateLocal"].dt.date
    player_stats["gameDateLocal_time"] = player_stats["gameDateLocal"].dt.time
    player_stats["gameDateLocal_year"] = player_stats["gameDateLocal"].dt.year
    player_stats["gameDateUTC_date"] = player_stats["gameDateUTC"].dt.date
    player_stats["gameDateUTC_time"] = player_stats["gameDateUTC"].dt.time
    player_stats["gameDateUTC_hour"] = player_stats["gameDateUTC"].dt.hour

    # Define rolling window for analysis
    analysis_start_year = datetime.today().year - cut_off

    # Filter rolling year
    player_stats = player_stats[
        player_stats["gameDateLocal_year"] >= analysis_start_year
    ]
    # Split dataframe for players whom total games in analysis year range meets 150 game or more
    total_games_last_nth_years_player_id_list = player_stats.groupby("personId")[
        "personId"
    ].count()
    total_games_last_nth_years_meets_threshold_player_id_list = (
        total_games_last_nth_years_player_id_list[
            total_games_last_nth_years_player_id_list >= 150
        ]
    )

    # The average games played in the dataset meets threshold
    refined_player_stats_data = player_stats[
        player_stats["personId"].isin(
            total_games_last_nth_years_meets_threshold_player_id_list.index
        )
    ].copy()

    # Create a full name ID to map player prop odds to player stats DF down the line
    # "P.J. Washington" → "PJWASHINGTON"
    refined_player_stats_data["fullNameId"] = refined_player_stats_data.apply(
        lambda x: x["firstName"].upper().replace(".", "").replace(" ", "")
        + x["lastName"].upper().replace(".", "").replace(" ", ""),
        axis=1,
    )
    # Filter Pre-season stats
    refined_player_stats_data = refined_player_stats_data[
        refined_player_stats_data["gameType"] != "Preseason"
    ].copy()
    # Create Next game date field
    refined_player_stats_data = refined_player_stats_data.sort_values(
        ["personId", "gameDateUTC"]
    )
    refined_player_stats_data["nextGameId"] = refined_player_stats_data.groupby(
        "personId"
    )["gameId"].shift(-1)
    refined_player_stats_data["nextGameDateUTC"] = refined_player_stats_data.groupby(
        "personId"
    )["gameDateUTC"].shift(-1)
    refined_player_stats_data["nextGameDateUTC_date"] = (
        refined_player_stats_data.groupby("personId")["gameDateUTC_date"].shift(-1)
    )
    refined_player_stats_data["nextGameDateUTC_time"] = (
        refined_player_stats_data.groupby("personId")["gameDateUTC_time"].shift(-1)
    )
    refined_player_stats_data["nextGameDateUTC_hour"] = (
        refined_player_stats_data.groupby("personId")["gameDateUTC_hour"].shift(-1)
    )

    # create Next Game Stats
    refined_player_stats_data["nextGamePoints"] = refined_player_stats_data.groupby(
        "personId"
    )["points"].shift(-1)
    refined_player_stats_data["nextGameAssists"] = refined_player_stats_data.groupby(
        "personId"
    )["assists"].shift(-1)
    refined_player_stats_data["nextGameThreePointersMade"] = (
        refined_player_stats_data.groupby("personId")["threePointersMade"].shift(-1)
    )
    refined_player_stats_data["nextGameReboundsTotal"] = (
        refined_player_stats_data.groupby("personId")["reboundsTotal"].shift(-1)
    )
    refined_player_stats_data["nextGameSteals"] = refined_player_stats_data.groupby(
        "personId"
    )["steals"].shift(-1)
    refined_player_stats_data["nextGameBlocks"] = refined_player_stats_data.groupby(
        "personId"
    )["blocks"].shift(-1)

    # 2 - Prep team bosxore stats
    team_stats_for_merge = team_stats[
        [
            "gameId",
            "teamName",
            "teamScore",
            "reboundsTotal",
            "fieldGoalsAttempted",
            "reboundsOffensive",
            "freeThrowsAttempted",
            "turnovers",
            "numMinutes",
        ]
    ].rename(
        columns={
            "teamName": "playerteamName",
            "teamScore": "playerteamTeamScore",
            "reboundsTotal": "playerteamTotalRebounds",
            "reboundsOffensive": "playerteamReboundsOffensive",
            "fieldGoalsAttempted": "playerteamfieldGoalsAttempted",
            "freeThrowsAttempted": "playerteamfreeThrowsAttempted",
            "turnovers": "playerteamTurnovers",
            "numMinutes": "playerteamNumMinutes",
        }
    )

    opp_team_stats_for_merge = team_stats[
        [
            "gameId",
            "teamName",
            "teamScore",
            "reboundsTotal",
            "reboundsOffensive",
            "reboundsDefensive",
            "blocks",
            "steals",
            "fieldGoalsAttempted",
            "fieldGoalsMade",
            "fieldGoalsPercentage",
            "turnovers",
            "freeThrowsAttempted",
            "plusMinusPoints",
        ]
    ].rename(
        columns={
            "teamName": "opponentteamName",
            "teamScore": "opponentteamScore",
            "reboundsTotal": "opponentTeamReboundsTotal",
            "reboundsOffensive": "opponentTeamReboundsOffensive",
            "reboundsDefensive": "opponentTeamReboundsDefensive",
            "fieldGoalsAttempted": "opponentTeamFieldGoalsAttempted",
            "fieldGoalsMade": "opponentTeamFieldGoalsMade",
            "fieldGoalsPercentage": "opponentTeamFieldGoalsPercentage",
            "blocks": "opponentTeamBlocks",
            "steals": "opponentTeamSteals",
            "turnovers": "opponentTeamTurnovers",
            "freeThrowsAttempted": "opponentTeamFreeThrowsAttempted",
            "plusMinusPoints": "opponentTeamPlusMinusPoints",
        }
    )

    # meagre team stats to respective games
    refined_player_stats_data = refined_player_stats_data.merge(
        team_stats_for_merge, on=["gameId", "playerteamName"]
    )
    refined_player_stats_data = refined_player_stats_data.merge(
        opp_team_stats_for_merge, on=["gameId", "opponentteamName"]
    )

    # 3- compute advanced stats
    # Compute Advanced Stats

    refined_player_stats_data["trueShooting"] = refined_player_stats_data["points"] / (
        2
        * (
            refined_player_stats_data["fieldGoalsAttempted"]
            + (0.44 * refined_player_stats_data["freeThrowsAttempted"])
        )
    )

    refined_player_stats_data["effectiveFieldGoal"] = (
        refined_player_stats_data["fieldGoalsMade"]
        + 0.5 * refined_player_stats_data["threePointersMade"]
    ) / refined_player_stats_data["fieldGoalsAttempted"]

    refined_player_stats_data["offensiveReboundingRate"] = (
        100
        * (
            refined_player_stats_data["reboundsOffensive"]
            * (refined_player_stats_data["playerteamNumMinutes"] / 5)
        )
        / (
            refined_player_stats_data["numMinutes"]
            * (
                refined_player_stats_data["playerteamTotalRebounds"]
                + refined_player_stats_data["opponentTeamReboundsTotal"]
            )
        )
    )

    refined_player_stats_data["totalReboundingRate"] = (
        100
        * (
            refined_player_stats_data["reboundsTotal"]
            * (refined_player_stats_data["playerteamNumMinutes"] / 5)
        )
        / (
            refined_player_stats_data["numMinutes"]
            * (
                refined_player_stats_data["playerteamTotalRebounds"]
                + refined_player_stats_data["opponentTeamReboundsTotal"]
            )
        )
    )

    refined_player_stats_data["usageRate"] = (
        100
        * (
            (
                refined_player_stats_data["fieldGoalsAttempted"]
                + 0.44 * refined_player_stats_data["freeThrowsAttempted"]
                + refined_player_stats_data["turnovers"]
            )
            * (refined_player_stats_data["playerteamNumMinutes"] / 5)
        )
        / (
            refined_player_stats_data["numMinutes"]
            * (
                refined_player_stats_data["playerteamfieldGoalsAttempted"]
                + 0.44 * refined_player_stats_data["playerteamfreeThrowsAttempted"]
                + refined_player_stats_data["playerteamTurnovers"]
            )
        )
    )

    # Opponent Team rebounding rate
    refined_player_stats_data["opponentTeamTotalReboundingRate"] = 100 * (
        refined_player_stats_data["opponentTeamReboundsTotal"]
        / (
            refined_player_stats_data["playerteamTotalRebounds"]
            + refined_player_stats_data["opponentTeamReboundsTotal"]
        )
    )

    # Oppenent team defensive rating
    # Estimate opponent possessions
    refined_player_stats_data["oppEstimatedPosessions"] = (
        refined_player_stats_data["playerteamfieldGoalsAttempted"]
        - refined_player_stats_data["playerteamReboundsOffensive"]
        + refined_player_stats_data["playerteamTurnovers"]
        + (0.44 * refined_player_stats_data["playerteamfreeThrowsAttempted"])
    )

    # opponent's Defensive rating
    refined_player_stats_data["opponentDefensiveRating"] = 100 * (
        refined_player_stats_data["playerteamTeamScore"]
        / refined_player_stats_data["oppEstimatedPosessions"]
    )

    # fill Nas with 0s
    refined_player_stats_data["trueShooting"] = refined_player_stats_data[
        "trueShooting"
    ].fillna(0)
    refined_player_stats_data["effectiveFieldGoal"] = refined_player_stats_data[
        "effectiveFieldGoal"
    ].fillna(0)
    refined_player_stats_data["totalReboundingRate"] = refined_player_stats_data[
        "totalReboundingRate"
    ].fillna(0)
    refined_player_stats_data["usageRate"] = refined_player_stats_data[
        "usageRate"
    ].fillna(0)
    refined_player_stats_data["opponentTeamTotalReboundingRate"] = (
        refined_player_stats_data["opponentTeamTotalReboundingRate"].fillna(0)
    )
    refined_player_stats_data["opponentDefensiveRating"] = refined_player_stats_data[
        "opponentDefensiveRating"
    ].fillna(0)
    refined_player_stats_data["opponentTeamPlusMinusPoints"] = (
        refined_player_stats_data["opponentTeamPlusMinusPoints"].fillna(0)
    )
    refined_player_stats_data["plusMinusPoints"] = refined_player_stats_data[
        "plusMinusPoints"
    ].fillna(0)

    # regular season games consists of specialized games (NBA Mexico City Game, NBA Paris Game) and early season tournaments. any games with Null fill wwith 0
    # playoffs will always have game series of 1-7 unlike regular, which may contain Nulls 0, 1,2 etc
    refined_player_stats_data["seriesGameNumber"] = refined_player_stats_data[
        "seriesGameNumber"
    ].fillna(0)

    # 4 - Trim columns
    refined_player_stats_data = refined_player_stats_data[
        [
            "firstName",
            "lastName",
            "fullNameId",
            "personId",
            "gameId",
            "gameDate",
            "gameDateLocal",
            "gameDateUTC",
            "gameDateLocal_date",
            "gameDateLocal_time",
            "gameDateLocal_year",
            "gameDateUTC_date",
            "gameDateUTC_time",
            "gameDateUTC_hour",
            "nextGameId",
            "nextGameDateUTC",
            "nextGameDateUTC_date",
            "nextGameDateUTC_time",
            "nextGameDateUTC_hour",
            "playerteamCity",
            "playerteamName",
            "opponentteamCity",
            "opponentteamName",
            "gameType",
            "gameLabel",
            "gameSubLabel",
            "seriesGameNumber",
            "win",
            "home",
            "numMinutes",
            "points",
            "assists",
            "blocks",
            "steals",
            "fieldGoalsAttempted",
            "fieldGoalsMade",
            "fieldGoalsPercentage",
            "threePointersAttempted",
            "threePointersMade",
            "threePointersPercentage",
            "freeThrowsAttempted",
            "freeThrowsMade",
            "freeThrowsPercentage",
            "reboundsDefensive",
            "reboundsOffensive",
            "reboundsTotal",
            "foulsPersonal",
            "turnovers",
            "plusMinusPoints",
            "usageRate",
            "trueShooting",
            "effectiveFieldGoal",
            "offensiveReboundingRate",
            "totalReboundingRate",
            "playerteamfieldGoalsAttempted",
            "playerteamfreeThrowsAttempted",
            "playerteamTurnovers",
            "opponentTeamBlocks",
            "opponentTeamSteals",
            "opponentTeamPlusMinusPoints",
            "opponentTeamReboundsOffensive",
            "opponentTeamReboundsDefensive",
            "opponentTeamTotalReboundingRate",
            "opponentDefensiveRating",
            "opponentTeamFieldGoalsAttempted",
            "opponentTeamFieldGoalsMade",
            "opponentTeamFieldGoalsPercentage",
            "nextGamePoints",
            "nextGameAssists",
            "nextGameThreePointersMade",
            "nextGameReboundsTotal",
            "nextGameSteals",
            "nextGameBlocks",
        ]
    ]

    return refined_player_stats_data


def build_features(
    df, stats_cols, event_date, sma_windows=[5, 10], ema_windows=[5, 10]
):
    """
    Build rolling features for player stats and contextual features.

    Args:
        df (pd.DataFrame): merged dataset (must include fullNameId, eventDate, stats cols)
        stats_cols (list): list of stats to generate SMA/EMA for
        event_date (str): use local datetime of event date
        sma_windows (list): window sizes for SMA
        ema_windows (list): window sizes for EMA

    Returns:
        pre_game_features (pd.DataFrame), training_dataset (pd.DataFrame)
    """

    df = df.copy()

    # Sort by player + date (important for rolling features!)
    df = df.sort_values(["fullNameId", event_date]).reset_index(drop=True)

    # Generate rolling features per player
    for stat in stats_cols:
        for w in sma_windows:
            df[f"{stat}_SMA{w}"] = df.groupby("fullNameId")[stat].transform(
                lambda x: x.shift().rolling(w, min_periods=1).mean()
            )
            # fill na with 0 for these stats
            df[f"{stat}_SMA{w}"] = df[f"{stat}_SMA{w}"].fillna(0)
        for w in ema_windows:
            df[f"{stat}_EMA{w}"] = df.groupby("fullNameId")[stat].transform(
                lambda x: x.shift().ewm(span=w, adjust=False).mean()
            )
            # fill na with 0 for these stats
            df[f"{stat}_EMA{w}"] = df[f"{stat}_EMA{w}"].fillna(0)

    # Context features
    # Calculate rest days
    df["restDays"] = (
        df.groupby("fullNameId")[event_date]
        .diff()  # difference in timestamps between consecutive games
        .dt.days  # convert to number of days
    )

    # Optional: Cap rest days if you want a max threshold (e.g., 14 days)
    df["restDays"] = df["restDays"].clip(upper=14)
    # assume cap rest day for Nulls, from analysis, these are mainly rookies with first game
    df["restDays"] = df["restDays"].fillna(14)

    return df


def outcome_label(row, line):
    stat_map = {
        "player_points": "points",
        "player_assists": "assists",
        "player_rebounds": "reboundsTotal",
        "player_threes": "threePointersMade",
        "player_steals": "steals",
        "player_blocks": "blocks",
    }

    stat_col = stat_map.get(row["market"])
    if stat_col is None or pd.isnull(row[stat_col]) or pd.isnull(line):
        return None  # no match or missing data

    value = row[stat_col]

    if row["overUnder"] == "Over":
        return 1 if value > line else 0
    elif row["overUnder"] == "Under":
        return 1 if value < line else 0
    else:
        return None


def predict_outcome_label(row, pred_stat_col, line):
    if pred_stat_col is None or pd.isnull(row[pred_stat_col]) or pd.isnull(line):
        return None  # no match or missing data

    value = row[pred_stat_col]

    if row["overUnder"] == "Over":
        return 1 if value > line else 0
    elif row["overUnder"] == "Under":
        return 1 if value < line else 0
    else:
        return None


def market_splitter(df, market_col="market"):
    market_splitter = {}
    for market in df["market"].unique():
        market_splitter[market] = df[df["market"] == market].copy()
    return market_splitter


def get_current_events(api_key, sport):
    # Get list of upcoming events (games)
    upcoming_events_url = f"https://api.the-odds-api.com/v4/sports/{sport}/events"
    upcoming_events_params = {"apiKey": api_key}

    response = requests.get(upcoming_events_url, params=upcoming_events_params)
    events = response.json()

    print(
        "Remaining Requests for the Month: {}".format(
            response.headers.get("x-requests-remaining")
        )
    )
    events_data = pd.DataFrame(events)

    if not events_data.empty:
        # Create game date and time column
        events_data["event_date"] = pd.to_datetime(
            events_data["commence_time"], utc=True
        ).dt.date
        events_data["event_time"] = pd.to_datetime(
            events_data["commence_time"], utc=True
        ).dt.time
        events_data["commence_time"] = pd.to_datetime(
            events_data["commence_time"], utc=True
        )

        # Save current events

        file_path = os.path.join(
            "odds",
            "daily",
            "events",
            f"daily_events_{sport}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.parquet",
        )
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        events_data.to_parquet(file_path, engine="pyarrow", index=False)

        return events_data
    else:
        print("No events found sport: {}".format(sport))
        return pd.DataFrame()


def get_current_player_props(df, sport, api_key):
    rows = []
    for EVENT_ID in df["id"]:
        # Player prop odds in each event(game)
        player_props_url = (
            f"https://api.the-odds-api.com/v4/sports/{sport}/events/{EVENT_ID}/odds"
        )
        player_props_params = {
            "apiKey": api_key,
            "regions": "us",  # US sportsbooks only
            "markets": "player_points,player_threes,player_assists,player_rebounds,player_blocks,player_steals",  # player props
            "oddsFormat": "decimal",
            "bookmakers": "fanduel",
        }
        # Events Request
        response = requests.get(player_props_url, params=player_props_params)
        data = response.json()

        # if response returns a single game, cast as list type
        if type(data) == dict:
            data = [data]

        for game in data:
            if "bookmakers" not in game or not game["bookmakers"]:
                print(f"No bookmakers found for event {game.get('id')}")
                continue  # skip this game

            for bookmaker in game["bookmakers"]:
                if bookmaker["key"] == "fanduel":
                    for market in bookmaker["markets"]:
                        for outcome in market["outcomes"]:
                            rows.append(
                                {
                                    "bookMaker": bookmaker["key"],
                                    "eventId": game["id"],
                                    "sport": game["sport_key"],
                                    "homeTeam": game["home_team"],
                                    "awayTeam": game["away_team"],
                                    "eventDateTime": pd.to_datetime(
                                        game["commence_time"], utc=True
                                    ),
                                    "eventDate": pd.to_datetime(
                                        game["commence_time"], utc=True
                                    ).date(),
                                    "eventTime": pd.to_datetime(
                                        game["commence_time"], utc=True
                                    ).time(),
                                    "eventDateTime_Hour": pd.to_datetime(
                                        game["commence_time"], utc=True
                                    ).hour,
                                    "timeStamp": market["last_update"],
                                    "player": outcome["description"],
                                    "fullNameId": (
                                        outcome["description"]
                                        .split(" ")[0]
                                        .upper()
                                        .replace(".", "")
                                        + "".join(outcome["description"].split(" ")[1:])
                                        .upper()
                                        .replace(".", "")
                                    ),
                                    "firstName": outcome["description"].split(" ")[0],
                                    "lastName": outcome["description"].split(" ")[1],
                                    "market": market["key"],
                                    "overUnder": outcome["name"],
                                    "line": outcome.get("point"),
                                    "odds": outcome["price"],
                                }
                            )

    player_props_data = pd.DataFrame(rows)
    if not player_props_data.empty:
        # Adjust team names to match kaggle datasets
        player_props_data["awayTeam"] = player_props_data["awayTeam"].apply(
            lambda x: " ".join(x.split(" ")[-2:])
            if "Trail" in x
            else "".join(x.split(" ")[-1:])
        )
        player_props_data["homeTeam"] = player_props_data["homeTeam"].apply(
            lambda x: " ".join(x.split(" ")[-2:])
            if "Trail" in x
            else "".join(x.split(" ")[-1:])
        )

        print(
            "Remaining Requests for the Month: {}".format(
                response.headers.get("x-requests-remaining")
            )
        )

        # Save current player props
        file_path = os.path.join(
            "odds",
            "daily",
            "player_props",
            f"daily_player_props_{sport}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.parquet",
        )
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        player_props_data.to_parquet(file_path, engine="pyarrow", index=False)

        return player_props_data
    else:
        print("No player prop mile stone found for selected bookmaker")
        return pd.DataFrame()


def create_historical_player_props_df(player_props, sport):
    """
    Takes list object of raw json player props output from The Odds API and flattens into proper plaer props DF
    for analysis. Saves datafram upon completion
    """

    historical_odds_row = []
    for wrapper in player_props:
        game = wrapper.get("data", {})
        if not game:  # skip if empty
            continue

        for bookmaker in game.get("bookmakers", []):
            if bookmaker["key"] == "fanduel":
                for market in bookmaker.get("markets", []):
                    for outcome in market.get("outcomes", []):
                        historical_odds_row.append(
                            {
                                "bookMaker": bookmaker["key"],
                                "eventId": game["id"],
                                "sport": game["sport_key"],
                                "homeTeam": game["home_team"],
                                "awayTeam": game["away_team"],
                                "eventDateTime": pd.to_datetime(
                                    game["commence_time"], utc=True
                                ),
                                "eventDate": pd.to_datetime(
                                    game["commence_time"], utc=True
                                ).date(),
                                "eventTime": pd.to_datetime(
                                    game["commence_time"], utc=True
                                ).time(),
                                "eventDateTime_Hour": pd.to_datetime(
                                    game["commence_time"], utc=True
                                ).hour,
                                "timeStamp": wrapper["timestamp"],
                                "player": outcome["description"],
                                "fullNameId": (
                                    outcome["description"]
                                    .split(" ")[0]
                                    .upper()
                                    .replace(".", "")
                                    + "".join(outcome["description"].split(" ")[1:])
                                    .upper()
                                    .replace(".", "")
                                ),
                                "firstName": outcome["description"].split(" ")[0],
                                "lastName": outcome["description"].split(" ")[1],
                                "market": market["key"],
                                "overUnder": outcome["name"],
                                "line": outcome.get("point"),
                                "odds": outcome["price"],
                            }
                        )
    historical_player_props_data = pd.DataFrame(historical_odds_row)
    historical_player_props_data = historical_player_props_data.sort_values(
        "timeStamp", ascending=False
    ).drop_duplicates(
        [
            "bookMaker",
            "eventId",
            "player",
            "fullNameId",
            "firstName",
            "lastName",
            "market",
            "overUnder",
        ]
    )

    # Adjust team names to match kaggle datasets
    historical_player_props_data["awayTeam"] = historical_player_props_data[
        "awayTeam"
    ].apply(
        lambda x: " ".join(x.split(" ")[-2:])
        if "Trail" in x
        else "".join(x.split(" ")[-1:])
    )
    historical_player_props_data["homeTeam"] = historical_player_props_data[
        "homeTeam"
    ].apply(
        lambda x: " ".join(x.split(" ")[-2:])
        if "Trail" in x
        else "".join(x.split(" ")[-1:])
    )

    # Save props
    historical_player_props_data.to_parquet(
        "odds/historical/player_props/historical_player_props_{}_{}.parquet".format(
            sport, datetime.now().strftime("%Y%m%d_%H%M%S")
        ),
        engine="pyarrow",
        index=False,
    )

    return historical_player_props_data


def fetch_historical_props_retry(
    event_id,
    commence_time,
    api_key,
    sport,
    bookmaker="fanduel",
    max_retry_hours=24,
    markets=None,
):
    """
    Tries to fetch historical player props by moving back in 1-hour increments
    until a snapshot is found or until 2 hours before commence time.

    Parameters:
        event_id (str): The event ID.
        commence_time (datetime): The game's commence time (UTC).
        api_key (str): Your API key.
        sport (str): Sport key, default from outer scope.
        bookmaker (str): Bookmaker key, default "fanduel".
        markets (list): List of markets, default common player stats.

    Returns:
        dict: The JSON response of the snapshot, or None if no snapshot found.
    """
    if markets is None:
        markets = [
            "player_points",
            "player_threes",
            "player_assists",
            "player_rebounds",
            "player_blocks",
            "player_steals",
        ]

    snapshot_time = commence_time.replace(minute=0, second=0, microsecond=0)
    cutoff_time = commence_time - timedelta(hours=2)  # stop 2 hours before game

    while snapshot_time >= cutoff_time:
        # convert to ISO8601 string with Z
        snapshot_time_str = snapshot_time.isoformat().replace("+00:00", "Z")

        url = f"https://api.the-odds-api.com/v4/historical/sports/{sport}/events/{event_id}/odds"
        params = {
            "apiKey": api_key,
            "regions": "us",
            "markets": ",".join(markets),
            "bookmakers": bookmaker,
            "date": snapshot_time_str,
        }

        response = requests.get(url, params=params)
        if response.status_code == 200 and response.json().get("data"):
            return response.json()  # snapshot found

        # Step back 1 hour
        snapshot_time -= timedelta(hours=1)
        sleep(1)

    # If loop completes without snapshot
    print(f"No snapshot found for event {event_id} within retry window.")
    return None


def get_historical_historical_events(api_key, sport, start_date, end_date):
    """
    Gets events through the odds api based on the specified start and end date provided
    data collected is automatocally saved upon completeion

    Date format
    datetime(2024, 10, 21, 23, 59, 59, tzinfo=timezone.utc) <-- odds-api looks at events on or before specified date
    datetime(2024, 10, 24, 23, 59, 59, tzinfo=timezone.utc)
    """

    json_file_path = "odds/historical/events/historical_events_{}_{}.json".format(
        sport, datetime.now().strftime("%Y%m")
    )
    os.makedirs(os.path.dirname(json_file_path), exist_ok=True)

    delta = timedelta(days=1)

    historical_events = []

    historical_current_date = start_date
    while historical_current_date <= end_date:
        date_str = historical_current_date.isoformat().replace("+00:00", "Z")
        print("Searching for events for {}".format(date_str))
        print("")

        # Get list of Historical events (games)
        historical_events_url = (
            f"https://api.the-odds-api.com/v4/historical/sports/{sport}/events"
        )
        historical_events_params = {"apiKey": api_key, "date": date_str}

        historical_response = requests.get(
            historical_events_url, params=historical_events_params
        )

        if historical_response.status_code == 200:
            historical_events.extend(historical_response.json()["data"])

            # Incremental saving
            if os.path.exists(json_file_path):
                with open(json_file_path, "r") as f:
                    try:
                        existing_data = json.load(f)
                    except json.JSONDecodeError:
                        existing_data = []
            else:
                existing_data = []

                # Extend existing data with new data
            existing_data.extend(historical_response.json()["data"])

            with open(json_file_path, "w") as f:
                json.dump(existing_data, f)

        else:
            print(f"Error for {date_str}: {historical_response.text}")
            print("")
        historical_current_date += delta

        sleep(2)

    # Create Data Frame and dedupe after
    historical_events_data = pd.DataFrame(historical_events)
    historical_events_data = historical_events_data.drop_duplicates("id")
    print(
        "Remaining Requests for the Month: {}".format(
            historical_response.headers.get("x-requests-remaining")
        )
    )

    historical_events_data["event_date"] = pd.to_datetime(
        historical_events_data["commence_time"], utc=True
    ).dt.date
    historical_events_data["event_time"] = pd.to_datetime(
        historical_events_data["commence_time"], utc=True
    ).dt.time
    historical_events_data["commence_time"] = pd.to_datetime(
        historical_events_data["commence_time"], utc=True
    )

    # adjust for games that start late which spill over into next utc day
    end_date += timedelta(days=1)
    # Filter for games in select timeframe
    mask = (historical_events_data["commence_time"] >= start_date) & (
        historical_events_data["commence_time"] <= end_date
    )
    historical_events_data = (
        historical_events_data[mask].copy().sort_values("commence_time")
    )

    # Save
    file_path = f"odds/historical/events/historical_events_{sport}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.parquet"
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    # Save excel file
    historical_events_data.to_parquet(file_path, engine="pyarrow", index=False)

    return historical_events_data


def get_historical_player_props(df, api_key, sport):
    file_path = (
        "odds/historical/player_props/historical_player_props_{}_{}.json".format(
            sport, datetime.now().strftime("%Y%m%d_%H%M%S")
        )
    )
    os.makedirs(os.path.dirname(file_path), exist_ok=True)

    historical_player_props = []
    for _, historical_game in df.iterrows():
        result = None

        # fix date to max time of day
        historical_odds_date = pd.to_datetime(
            historical_game["commence_time"], utc=True
        )
        historical_odds_date_str = (
            historical_odds_date.replace(hour=23, minute=59, second=59)
            .isoformat()
            .replace("+00:00", "Z")
        )
        print(
            "Searching for Players props for {}, event id:".format(
                historical_odds_date_str, historical_game["id"]
            )
        )
        print("")

        # API call
        historical_event_url = f"https://api.the-odds-api.com/v4/historical/sports/{sport}/events/{historical_game['id']}/odds?"
        historical_params = {
            "apiKey": api_key,
            "regions": "us",
            "markets": "player_points,player_threes,player_assists,player_rebounds,player_blocks,player_steals",
            "bookmakers": "fanduel",
            "date": historical_odds_date_str,
        }

        historical_response = requests.get(
            historical_event_url, params=historical_params
        )

        if historical_response.status_code == 200:
            result = historical_response.json()
            historical_player_props.append(result)
            print(
                "Remaining Requests for the Month:",
                historical_response.headers.get("x-requests-remaining"),
            )
        else:
            retry_fetch = fetch_historical_props_retry(
                event_id=historical_game["id"],
                commence_time=historical_odds_date,
                api_key=api_key,
                sport=sport,
            )
            if retry_fetch is not None:
                result = retry_fetch
                historical_player_props.append(result)
                print("Retry succeeded for", historical_game["id"])
            else:
                print(
                    f"Error for event {historical_game['id']} at {historical_odds_date_str}: {historical_response.text}"
                )

        # Incremental saving
        if result:
            if os.path.exists(file_path):
                with open(file_path, "r") as f:
                    try:
                        existing_data = json.load(f)
                    except json.JSONDecodeError:
                        existing_data = []
            else:
                existing_data = []

            existing_data.append(result)

            with open(file_path, "w") as f:
                json.dump(existing_data, f)

            sleep(2)

    return historical_player_props


def clean_props_with_boxscore(df, minuts_played_col="numMinutes", game_type_col=None):
    # remove player records where player recorded no minutes, These may be due to Injury, Coaches benching key players, load management, trades, etc.
    df = df[df[minuts_played_col].notnull()].copy()

    # Remove infinite values - This typically occurs when a player was recorded with an insignificant amount of minutes
    # EXAMPLE: Jalen brunson 3/3/24, comuting advanced stats on this has wonky results
    mask_inf = df.isin([np.inf, -np.inf]).any(axis=1)
    df = df[~mask_inf].copy()

    # remove null cols, these columns hold no statistical value at this point
    nulls = pd.isnull(df).sum()
    null_cols = nulls[nulls > 0].index
    valid_cols = nulls[~nulls.index.isin(nulls)].index
    df = df[valid_cols]

    if game_type_col != None:
        # string enocoding for game types
        encoder_path = os.path.join(
            os.path.dirname(__file__),
            "..",
            "models/basketball_nba/nba_game_type_label_encoder.pkl",
        )
        le = joblib.load(os.path.abspath(encoder_path))
        df[game_type_col] = le.fit_transform(df[game_type_col])

    return df


def ml_feature_selection(sport="basketball_nba", market=None):
    if sport == "basketball_nba":
        if market == "player_points":
            features = [
                "points_EMA10",
                "points_SMA10",
                "points_SMA5",
                "points_EMA5",
                "usageRate_EMA10",
                "usageRate_SMA10",
                "usageRate_EMA5",
                "usageRate_SMA5",
                "numMinutes_EMA10",
                "numMinutes_SMA10",
                "freeThrowsPercentage_EMA10",
                "numMinutes_SMA5",
                "numMinutes_EMA5",
                "turnovers_EMA10",
                "freeThrowsPercentage_SMA10",
                "turnovers_SMA10",
                "turnovers_EMA5",
                "turnovers_SMA5",
                "assists_EMA10",
                "freeThrowsPercentage_EMA5",
                "freeThrowsPercentage_SMA5",
                "assists_SMA10",
                "assists_EMA5",
                "assists_SMA5",
                "threePointersAttempted_EMA10",
                "threePointersAttempted_SMA10",
                "threePointersAttempted_SMA5",
                "threePointersAttempted_EMA5",
                "threePointersMade_SMA10",
                "threePointersMade_EMA10",
            ]

        elif market == "player_assists":
            features = [
                "assists_EMA10",
                "assists_SMA10",
                "assists_EMA5",
                "assists_SMA5",
                "turnovers_EMA10",
                "turnovers_SMA10",
                "turnovers_EMA5",
                "turnovers_SMA5",
                "points_SMA10",
                "usageRate_EMA10",
                "points_EMA10",
                "usageRate_SMA10",
                "usageRate_SMA5",
                "points_SMA5",
                "usageRate_EMA5",
                "points",
                "points_EMA5",
                "numMinutes_EMA10",
                "numMinutes_SMA10",
                "numMinutes_SMA5",
                "threePointersAttempted_SMA5",
                "numMinutes_EMA5",
                "threePointersAttempted_SMA10",
                "threePointersAttempted_EMA10",
                "threePointersAttempted_EMA5",
                "freeThrowsPercentage_SMA10",
                "freeThrowsPercentage_EMA10",
                "threePointersMade_SMA10",
                "threePointersMade_SMA5",
                "threePointersMade_EMA10",
            ]

        elif market == "player_threes":
            features = [
                "threePointersAttempted_SMA10",
                "threePointersAttempted_EMA10",
                "threePointersAttempted_SMA5",
                "threePointersAttempted_EMA5",
                "threePointersMade_SMA10",
                "threePointersMade_EMA10",
                "threePointersMade_SMA5",
                "threePointersMade_EMA5",
                "points_SMA5",
                "points_EMA10",
                "points_SMA10",
                "numMinutes_SMA5",
                "points_EMA5",
                "numMinutes_EMA10",
                "numMinutes_SMA10",
                "numMinutes_EMA5",
                "reboundsOffensive_EMA10",
                "assists_SMA5",
                "assists_EMA5",
                "reboundsOffensive_SMA10",
                "assists_EMA10",
                "reboundsOffensive_EMA5",
                "assists_SMA10",
                "usageRate_EMA10",
                "usageRate_SMA5",
                "usageRate_EMA5",
                "usageRate_SMA10",
                "turnovers_EMA5",
                "turnovers_SMA5",
                "turnovers_EMA10",
            ]

        elif market == "player_rebounds":
            features = [
                "reboundsTotal_EMA10",
                "reboundsTotal_SMA10",
                "reboundsTotal_EMA5",
                "reboundsTotal_SMA5",
                "totalReboundingRate_SMA10",
                "totalReboundingRate_EMA10",
                "totalReboundingRate_EMA5",
                "totalReboundingRate_SMA5",
                "reboundsOffensive_SMA10",
                "reboundsDefensive",
                "reboundsOffensive_EMA10",
                "reboundsOffensive_SMA5",
                "totalReboundingRate",
                "reboundsOffensive_EMA5",
                "fieldGoalsPercentage_SMA10",
                "blocks_EMA10",
                "fieldGoalsPercentage_EMA10",
                "blocks_SMA10",
                "blocks_SMA5",
                "numMinutes_SMA10",
                "numMinutes_EMA10",
                "blocks_EMA5",
                "points_SMA10",
                "numMinutes_SMA5",
                "points_EMA10",
                "numMinutes_EMA5",
                "points_SMA5",
                "fieldGoalsPercentage_SMA5",
                "points_EMA5",
                "fieldGoalsPercentage_EMA5",
            ]

        elif market == "player_steals":
            features = [
                "steals_EMA10",
                "steals_SMA10",
                "steals_EMA5",
                "opponentTeamFieldGoalsPercentage_SMA10",
                "opponentTeamBlocks_EMA5",
                "opponentTeamFieldGoalsMade_SMA10",
                "steals_SMA5",
                "opponentTeamBlocks_EMA10",
                "opponentTeamFieldGoalsMade_EMA10",
                "opponentTeamBlocks_SMA5",
                "opponentTeamFieldGoalsPercentage_EMA10",
                "opponentTeamFieldGoalsMade_SMA5",
                "opponentTeamBlocks_SMA10",
                "opponentTeamFieldGoalsMade_EMA5",
                "opponentTeamPlusMinusPoints_SMA5",
                "opponentTeamReboundsDefensive",
                "opponentTeamPlusMinusPoints_SMA10",
                "opponentTeamFieldGoalsPercentage_SMA5",
                "opponentTeamTotalReboundingRate",
                "opponentTeamReboundsOffensive_SMA10",
                "plusMinusPoints_SMA5",
                "opponentTeamPlusMinusPoints_EMA10",
                "opponentTeamFieldGoalsPercentage_EMA5",
                "totalReboundingRate_SMA10",
                "opponentTeamFieldGoalsAttempted_SMA5",
                "plusMinusPoints_EMA10",
                "threePointersMade_SMA5",
                "opponentDefensiveRating_SMA5",
                "assists_EMA5",
                "opponentTeamFieldGoalsAttempted_EMA10",
            ]

        elif market == "player_blocks":
            features = [
                "blocks_EMA10",
                "blocks_SMA10",
                "blocks_SMA5",
                "blocks_EMA5",
                "assists_EMA5",
                "assists_EMA10",
                "assists_SMA5",
                "assists_SMA10",
                "opponentTeamTotalReboundingRate",
                "trueShooting_EMA10",
                "trueShooting_EMA5",
                "turnovers_EMA5",
                "effectiveFieldGoal_EMA10",
                "steals_SMA10",
                "turnovers_EMA10",
                "turnovers_SMA5",
                "effectiveFieldGoal_EMA5",
                "points",
                "usageRate_SMA5",
                "usageRate_EMA5",
                "usageRate_EMA10",
                "restDays",
                "trueShooting_SMA10",
                "steals_EMA10",
                "opponentTeamTotalReboundingRate_EMA10",
                "usageRate_SMA10",
                "opponentTeamTotalReboundingRate_SMA5",
                "trueShooting_SMA5",
                "opponentTeamFieldGoalsPercentage",
                "opponentTeamFieldGoalsAttempted_SMA10",
            ]
        else:
            print("No market provided, no custom feature selection was done to subset")

        return features

    else:
        print(
            "sport title: {}, not known , no custom feature selection was done to subset".format(
                sport
            )
        )
        return None


def evaluate_predictions(
    df,
    line_col="line",
    predicted_line_col="predictedLineOutCome",
    actual_stat_col="outcomeStat",
    prop_type_col="overUnder",
    result_col="predictionCorrect",
):
    """
    Compare predicted outcome vs actual results and return a new column 'isCorrect'.

    predictedLineOutCome:
        1 → Over predicted
        0 → Under predicted
    """
    if prop_type_col == "overUnder":
        # Safety: ensure needed columns exist
        required = ["predictedLineOutCome", "line", "outcomeStat", "overUnder"]
        for col in required:
            if col not in df.columns:
                raise ValueError(f"Missing required column: {col}")

        # Apply comparison logic
        def check_correct(row):
            if (
                pd.isnull(row[predicted_line_col])
                or pd.isnull(row[line_col])
                or pd.isnull(row[actual_stat_col])
            ):
                return np.nan
            if (row[predicted_line_col] == 1) and (
                row[prop_type_col] == "Over"
            ):  # predicted Over
                return row[actual_stat_col] > row[line_col]
            elif row[predicted_line_col] == 0 and (
                row[prop_type_col] == "Over"
            ):  # predicted Over
                return row[actual_stat_col] < row[line_col]
            elif (row[predicted_line_col] == 1) and (
                row[prop_type_col] == "Under"
            ):  # predicted Under
                return row[actual_stat_col] < row[line_col]
            elif row[predicted_line_col] == 0 and (
                row[prop_type_col] == "Under"
            ):  # predicted Under
                return row[actual_stat_col] > row[line_col]
            else:
                return np.nan

        df[result_col] = df.apply(check_correct, axis=1)
        return df


def format_email_body(df):
    html_parts = []

    for event_id in df["eventId"].unique():
        game_df = df[df["eventId"] == event_id].copy()
        match_up = "{} at {}".format(
            game_df["awayTeam"].iloc[0], game_df["homeTeam"].iloc[0]
        )

        game_df["take"] = game_df.apply(
            lambda row: (
                "Over"
                if row["overUnder"] == "Over" and row["predictedLineOutCome"] == 1
                else "Under"
                if row["overUnder"] == "Over" and row["predictedLineOutCome"] == 0
                else "Under"
                if row["overUnder"] == "Under" and row["predictedLineOutCome"] == 1
                else "Over"
            ),
            axis=1,
        )

        game_df = game_df[["player", "market", "line", "take", "odds", "probability"]]

        # Append HTML for this event
        html_parts.append(f"""
            <h3>{match_up}</h3>
            {game_df.to_html(index=False, border=0)}
            <br>
        """)

    return "\n".join(html_parts)


def email_betting_odds(df):
    msg = EmailMessage()
    recipients = [
        "javolwilloughby@gmail.com",
        "Guillermoulloa14@gmail.com",
        "Gpinales15@gmail.com",
    ]

    msg["Subject"] = "NBA Fanduel Odds"
    msg["From"] = EMAIL_ADDRESS
    msg["To"] = ", ".join(recipients)

    # Plain text + HTML version
    html = f"""
			<html>
			  <body>
			    <p>Here are today's fanduel player props prediction (as of {datetime.now().strftime("%m/%d/%Y %H:%M")}):</p>
			    {format_email_body(df)}
			  </body>
			</html>
			"""
    msg.add_alternative(html, subtype="html")
    with smtplib.SMTP_SSL("smtp.gmail.com", 465) as smtp:
        smtp.login(EMAIL_ADDRESS, EMAIL_PWD)
        smtp.send_message(msg, to_addrs=recipients)


def collect_and_merge_predictions_and_outcomes(stats_df, sport):
    try:
        # collect all predicted files
        path = "./outcomes/daily/predictions/basketball_nba"
        files = [f for f in os.listdir(path) if os.path.isfile(os.path.join(path, f))]
        files = [f for f in files if f.endswith(".parquet")]

        predictions = pd.DataFrame()
        for file in files:
            file_df = pd.read_parquet(path + "/" + file, engine="pyarrow")
            file_df["file_name"] = file
            file_df["file_date"] = file[:-24]

            predictions = pd.concat([file_df, predictions])

        predictions = (
            predictions.sort_values("file_name", ascending=False)
            .drop_duplicates(["eventId", "personId", "market", "overUnder"])
            .reset_index(drop=True)
        )
        predictions["predictedStat"] = (
            predictions[
                [
                    "predictedPoints",
                    "predictedAssists",
                    "predictedThrees",
                    "predictedRebounds",
                    "predictedSteals",
                    "predictedBlocks",
                ]
            ]
            .bfill(axis=1)
            .iloc[:, 0]
        )
        predictions = predictions.drop(
            [
                "predictedPoints",
                "predictedAssists",
                "predictedThrees",
                "predictedRebounds",
                "predictedSteals",
                "predictedBlocks",
                "file_name",
                "file_date",
                "nextGamePoints",
                "nextGameAssists",
                "nextGameThreePointersMade",
                "nextGameReboundsTotal",
                "nextGameSteals",
                "nextGameBlocks",
            ],
            axis=1,
        )
        # covert datatype for potential merge
        predictions["personId"] = predictions["personId"].astype("int32")

        # Stats columns you want to break out
        stats_cols = [
            "points",
            "threePointersMade",
            "assists",
            "reboundsTotal",
            "steals",
            "blocks",
        ]

        # Melt the DataFrame
        long_df = stats_df.melt(
            id_vars=["personId", "gameDateUTC"],
            value_vars=stats_cols,
            var_name="market",
            value_name="outcomeStat",
        )

        long_df["market"] = long_df["market"].replace(
            {
                "points": "player_points",
                "threePointersMade": "player_threes",
                "assists": "player_assists",
                "reboundsTotal": "player_rebounds",
                "steals": "player_steals",
                "blocks": "player_blocks",
            }
        )
        long_df = long_df.rename(columns={"gameDateUTC": "eventDateTime"})

        predictions_outcomes = pd.merge_asof(
            predictions.sort_values("eventDateTime"),
            long_df.sort_values("eventDateTime"),
            left_on="eventDateTime",
            right_on="eventDateTime",
            by=["personId", "market"],
            tolerance=pd.Timedelta("2h"),
            direction="nearest",
        )

        predictions_outcomes = evaluate_predictions(predictions_outcomes)

        return predictions_outcomes

    except Exception as e:
        print(f"❌ merge_predictions() failed: {e}")
        # Optionally log error stack trace
        import traceback

        traceback.print_exc()
        # Return failure flag
        return False


def check_sports_schedule(sport):
    if sport == "basketball_nba":
        # collect all predicted files
        path = "./nba_data"
        files = [f for f in os.listdir(path) if os.path.isfile(os.path.join(path, f))]
        schedule_files = [f for f in files if "schedule" in f.lower()]
        latest_schedule = sorted(schedule_files, reverse=True)[0]
        nba_schedule = pd.read_csv(os.path.join(path, latest_schedule))
        nba_schedule = nba_schedule[nba_schedule["gameLabel"] != "Preseason"].copy()

        nba_schedule["gameDate"] = (
            nba_schedule["gameDateTimeEst"].apply(parse_game_date).dt.date
        )
        todays_schedule = nba_schedule[
            nba_schedule["gameDate"]
            == datetime.now(ZoneInfo("America/New_York")).date()
        ]

        return todays_schedule.empty


def get_latest_model_in_log(sport, stat_type, model_type):
    if stat_type == "points":
        market = "player_points"
        target = "nextGamePoints"

    elif stat_type == "assists":
        market = "player_assists"
        target = "nextGameAssists"

    elif stat_type == "threes":
        market = "player_threes"
        target = "nextGameThreePointersMade"

    elif stat_type == "rebounds":
        market = "player_rebounds"
        target = "nextGameReboundsTotal"

    elif stat_type == "blocks":
        market = "player_blocks"
        target = "nextGameBlocks"

    elif stat_type == "steals":
        market = "player_steals"
        target = "nextGameSteals"

    # If folder or no files exist in XGBOOST calibration go to base model
    if os.path.exists(f"./models/{sport}/xgboost_regressor_{model_type}_calibrations"):
        # path exists, get latest calibrated xgboost models
        path = f"./models/{sport}/xgboost_regressor_{model_type}_calibrations"
        models = [f for f in os.listdir(path) if os.path.isfile(os.path.join(path, f))]
        models_path = [f for f in models if stat_type.lower() in f.lower()]
        models_path.sort(reverse=True)  # Sort for most recent
    else:
        models_path = []

    if models_path != []:
        model_path = models_path[0]  # grab latest model
        model_path = path + "/" + model_path
        current_xgb_model = joblib.load(model_path)

    else:
        # start with base model
        path = f"./models/{sport}/xgboost_regressor_base"
        models = [f for f in os.listdir(path) if os.path.isfile(os.path.join(path, f))]
        model_path = [f for f in models if stat_type.lower() in f.lower()][0]
        model_path = path + "/" + model_path

        # Assume booster file does not exist the same base name as the pkl file
        current_xgb_model = joblib.load(model_path)

    return market, target, current_xgb_model, model_path


def collect_all_player_props():
    # collect Historical props
    historical_path = f"./odds/historical/player_props/"
    daily_path = f"./odds/daily/player_props/"

    all_files = []
    for path in [historical_path, daily_path]:
        files = [f for f in os.listdir(path) if os.path.isfile(os.path.join(path, f))]
        files = [path + "/" + f for f in files if f.endswith(".parquet")]
        all_files.extend(files)

    # Remove old obselete file
    all_files = [
        f
        for f in all_files
        if "historical_player_props_basketball_nba_20250924_193509.parquet" not in f
    ]

    # Dump props into dataframe from files
    props = pd.DataFrame()
    for file in all_files:
        file_df = pd.read_parquet(file, engine="pyarrow")
        props = pd.concat([file_df, props])

    props = (
        props.sort_values("timeStamp", ascending=False)
        .drop_duplicates(["eventId", "fullNameId", "market", "overUnder"])
        .reset_index(drop=True)
    )

    return props


def create_isotonic_regressor_calibrator_model(
    stats, props_df, market, target, xgb_progressive_model, sport, model_type, stat_type
):
    """This functions creates ML calebration model based off the the parameters passed in.
    This function should be used inside of each model create function only to allow smooth implememntation"""

    props_and_box = pd.merge_asof(
        props_df.sort_values("eventDateTime"),
        stats.sort_values("nextGameDateUTC"),
        left_on="eventDateTime",
        right_on="nextGameDateUTC",
        by=["fullNameId"],
        tolerance=pd.Timedelta("2h"),
        direction="nearest",
    )

    # data preperation
    props_and_box["home"] = props_and_box.apply(
        lambda row: 1 if row["homeTeam"] == row["playerteamName"] else 0, axis=1
    )
    props_and_box = clean_props_with_boxscore(props_and_box, "numMinutes", "gameType")
    props_and_box = props_and_box[
        (props_and_box["market"] == market)
        & (
            props_and_box["eventDate"]
            >= (datetime.now(tz=pytz.utc) - relativedelta(weeks=8)).date()
        )
    ].copy()

    # Start calibration
    features = ml_feature_selection(sport=sport, market=market)
    X = props_and_box[features]
    y = props_and_box[target]

    prediction = xgb_progressive_model.predict(X)  # predictions using recent model

    calibrator = IsotonicRegression(out_of_bounds="clip")
    calibrator.fit(prediction, y)
    bundle = {"xgb_progressive_model": xgb_progressive_model, "calibrator": calibrator}

    # Save Isotonic Bundle
    os.makedirs(
        f"./models/{sport}/xgboost_regressor_progressive_isotonic_calibrations",
        exist_ok=True,
    )
    joblib.dump(
        bundle,
        f"./models/{sport}/xgboost_regressor_progressive_isotonic_calibrations/xgb_progessive_isotonic_{stat_type}_model_{datetime.now().date().strftime('%Y%m%d')}.pkl",
    )

    print(
        f"✅ Isotonic calibration model created and saved for {sport} - {market} ({stat_type})."
    )


def update_xgb_progressive_models(
    stats_df,
    props_df,
    market,
    stat_type,
    model_type,
    target,
    sport,
    current_xgb_model,
    og_model_path,
    force_start_date=None,
):
    # Select last week’s data (if applicable)
    if force_start_date == None:
        if os.path.exists(
            f"./models/{sport}/model_update_tracker.csv"
        ):  # if tracker file exists, check there first
            tracker_file = pd.read_csv(f"./models/{sport}/model_update_tracker.csv")
            if (
                tracker_file[tracker_file["model"] == f"progressive_{stat_type}"][
                    "last_update"
                ].iloc[0]
                != None
            ):  # Check if value exists for model
                tracker_file["last_update"] = pd.to_datetime(
                    tracker_file["last_update"]
                ).dt.date
                start_date = tracker_file[
                    tracker_file["model"] == f"progressive_{stat_type}"
                ]["last_update"].iloc[0]
            else:
                start_date = datetime.now(tz=pytz.utc).date() - relativedelta(days=1)
        else:
            start_date = datetime.now(tz=pytz.utc).date() - relativedelta(days=1)
    else:
        start_date = force_start_date

    # Step 1 - data preperation
    stats_df = stats_df[stats_df["nextGameDateUTC"].notna()].copy()
    props_and_box = pd.merge_asof(
        props_df.sort_values("eventDateTime"),
        stats_df.sort_values("nextGameDateUTC"),
        left_on="eventDateTime",
        right_on="nextGameDateUTC",
        by=["fullNameId"],
        tolerance=pd.Timedelta("2h"),
        direction="nearest",
    )

    props_and_box["home"] = props_and_box.apply(
        lambda row: 1 if row["homeTeam"] == row["playerteamName"] else 0, axis=1
    )
    props_and_box = clean_props_with_boxscore(props_and_box, "numMinutes", "gameType")
    props_and_box = props_and_box[props_and_box["eventDate"] >= start_date]
    market_df = props_and_box[props_and_box["market"] == market]

    if not market_df.empty:
        # Step 2 - ML prepping
        selected_features = ml_feature_selection(sport, market)
        X = market_df[selected_features]
        y = market_df[target]

        x_train, x_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=10
        )  # Split, train, test

        # Current model predictions
        current_y_pred = current_xgb_model.predict(x_test)
        current_rmse = np.sqrt(metrics.mean_squared_error(y_test, current_y_pred))

        # New model predictions
        update_xgb_model = copy.deepcopy(
            current_xgb_model
        )  # <-- deep copy to avoid having the same model in memory
        update_xgb_model = update_xgb_model.fit(
            x_train, y_train, xgb_model=current_xgb_model.get_booster()
        )
        update_y_pred = update_xgb_model.predict(x_test)
        update_rmse = np.sqrt(metrics.mean_squared_error(y_test, update_y_pred))

        # Step 3 - compare old and new model, save output
        if update_rmse <= current_rmse:
            print("New model Accepted")
            final_model = update_xgb_model.fit(
                X, y, xgb_model=update_xgb_model.get_booster()
            )  # Update progressive learner

            # Save
            os.makedirs(
                f"./models/{sport}/xgboost_regressor_{model_type}_calibrations",
                exist_ok=True,
            )
            joblib.dump(
                final_model,
                f"./models/{sport}/xgboost_regressor_{model_type}_calibrations/xgb_{stat_type}_model_{datetime.now().date().strftime('%Y%m%d')}.pkl",
            )

            # Save latest date a model was last upgraded on
            if os.path.exists(f"./models/{sport}/model_update_tracker.csv"):
                latest_fitted_date = market_df["eventDateTime"].max()
                tracker = pd.read_csv(f"./models/{sport}/model_update_tracker.csv")
                tracker.loc[
                    tracker["model"] == f"progressive_{stat_type}", "last_update"
                ] = latest_fitted_date
                tracker.to_csv(
                    f"./models/{sport}/model_update_tracker.csv", index=False
                )
            else:
                latest_fitted_date = market_df["eventDateTime"].max()
                # create tracker file
                model_stats = [
                    "progressive_points",
                    "progressive_assists",
                    "progressive_threes",
                    "progressive_rebounds",
                    "progressive_steals",
                    "progressive_blocks",
                ]
                tracker = pd.DataFrame({"model": model_stats})
                tracker["last_update"] = None
                tracker.loc[
                    tracker["model"] == f"progressive_{stat_type}", "last_update"
                ] = latest_fitted_date
                tracker.to_csv(
                    f"./models/{sport}/model_update_tracker.csv", index=False
                )

            # Step 4 - update Isotonic regression
            create_isotonic_regressor_calibrator_model(
                stats=stats_df,
                props_df=props_df,
                market=market,
                target=target,
                sport=sport,
                xgb_progressive_model=final_model,
                model_type=model_type,
                stat_type=stat_type,
            )

    else:
        print(
            f"Dataset for {sport}, {market} for date start at {start_date} is empty... No models updated for today"
        )


def get_active_model_for_production(sport, stat_type):
    """Active models are the models that performs the best on a weekly basis per stat,
    will be found in teh root model-sport directory"""

    # If folder or no files exist in XGBOOST calibration go to base model
    if os.path.exists(f"./models/{sport}"):
        path = f"./models/{sport}"
        models = [f for f in os.listdir(path) if os.path.isfile(os.path.join(path, f))]
        models_path = [f for f in models if stat_type.lower() in f.lower()]

        if models_path != []:
            model_path = models_path[0]
            model_path = path + "/" + model_path
            production_model = joblib.load(model_path)

            return (production_model, model_path)
        else:
            print("No active models in production path")
            return None
    else:
        print("No active models in production path")
        return None


def prop_model_evaluation(
    model, model_type, model_path, stat_type, props_box_df, sport
):
    """
    Returns predictions and metadata for a given model_type/stat_type combination
    """

    if stat_type == "points":
        market = "player_points"
        target = "nextGamePoints"

    elif stat_type == "assists":
        market = "player_assists"
        target = "nextGameAssists"

    elif stat_type == "threes":
        market = "player_threes"
        target = "nextGameThreePointersMade"

    elif stat_type == "rebounds":
        market = "player_rebounds"
        target = "nextGameReboundsTotal"

    elif stat_type == "blocks":
        market = "player_blocks"
        target = "nextGameBlocks"

    elif stat_type == "steals":
        market = "player_steals"
        target = "nextGameSteals"

    market_df = props_box_df.copy()
    market_df = market_df[market_df["market"] == market]

    features = ml_feature_selection(sport=sport, market=market)
    X = market_df[features]

    if "isotonic" in model_path.lower():
        # unbundle isotonic
        calibrator = model["calibrator"]
        model = model["xgb_progressive_model"]
        model.set_params(n_jobs=1)

        # Make Predictions
        raw_preds = model.predict(X)
        market_df["modelPrediction"] = calibrator.predict(raw_preds)
    else:
        # Make Predictions
        model.set_params(n_jobs=1)
        market_df["modelPrediction"] = model.predict(X)

    del model
    market_df["modelType"] = model_type
    market_df["modelPath"] = model_path
    market_df["marketType"] = market

    return market_df


def replace_active_models(model_path_list, sport, stat_list):
    # Get Active models in production path
    paths_to_remove = []
    for stat in stat_list:
        active_path = get_active_model_for_production(sport=sport, stat_type=stat)
        paths_to_remove.append(active_path)

    # Remove models in production path
    for path in paths_to_remove:
        if path is not None:
            os.remove(path[1])
            print(f"Deleted: {path[1]}")

    # Copy best models to prodiction path
    for source_path in model_path_list:
        shutil.copy(source_path, f"./models/{sport}")
        print(f"{source_path} moved to production path")


# def get_player_images(fullNameId):
#     path = "./player_images/"
#     image_folder = os.listdir(path)
#     fullNameId = fullNameId.upper().replace(".", "").replace(" ", "").replace("-", "")

#     image_list = []
#     for file_name in image_folder:
#         full_name_id = (
#             file_name[:-4].upper().replace(".", "").replace(" ", "").replace("-", "")
#         )
#         file_path = path + file_name
#         path_and_name_id = [full_name_id, file_path]
#         image_list.append(path_and_name_id)

#     image_df = pd.DataFrame(data=image_list, columns=["fullNameId", "path"])
#     image = image_df[image_df["fullNameId"] == fullNameId]

#     if image.empty:
#         return image_df[image_df["fullNameId"] == "DOWNLOAD"]["path"].iloc[0]
#     else:
#         return image["path"].iloc[0]
# 	else:
# 		return image['path'].iloc[0]
