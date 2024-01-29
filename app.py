import boto3
import os
import numpy as np
import pandas as pd
import pickle
import psycopg2
import warnings

warnings.filterwarnings("ignore")


from flask import Flask, render_template, request
from helper_dicts import logo_dict, file_name_dict, team_name_dict

app = Flask(__name__)


def create_db_connection():
    return psycopg2.connect(
        database=os.getenv("AWS_PSQL_DB"),
        user=os.getenv("AWS_PSQL_USER"),
        password=os.getenv("AWS_PSQL_PASSWORD"),
        host=os.getenv("AWS_PSQL_HOST"),
        port=os.getenv("AWS_PSQL_PORT"),
    )


s3 = boto3.resource("s3")

bucket = s3.Bucket(os.getenv("AWS_S3_MODEL_SRC"))

current_file = max(bucket.objects.all(), key=lambda obj: obj.key).key

model_pickle = pickle.loads(
    s3.Bucket(os.getenv("AWS_S3_MODEL_SRC")).Object(current_file).get()["Body"].read()
)

model, model_metadata = model_pickle[0]
old_school_model, old_school_metadata = model_pickle[1]
modern_model, modern_metadata = model_pickle[2]


@app.route("/")
def display_games():
    current_file = max(bucket.objects.all(), key=lambda obj: obj.key).key

    model_pickle = pickle.loads(
        s3.Bucket("mlb-prediction-model").Object(current_file).get()["Body"].read()
    )

    model, model_metadata = model_pickle[0]
    old_school_model, old_school_metadata = model_pickle[1]
    modern_model, modern_metadata = model_pickle[2]

    predict_games(model)

    conn = create_db_connection()
    cursor = conn.cursor()

    sql = f"select * from games where predicted_winner is not null;"

    cursor.execute(sql)

    rows = cursor.fetchall()

    column_names = [desc[0] for desc in cursor.description]

    df = pd.DataFrame(rows, columns=column_names)

    games_correct = 0
    games_attempted = 0
    future_games = 0
    processed_data = []

    for _, row in df.iterrows():
        future_game = pd.isna(row["winning_team"])

        games_attempted += 1
        if future_game:
            future_games += 1

        game = {
            "team1": row["home_team_name"],
            "team2": row["away_team_name"],
            "prediction": row["home_team_name"]
            if row["predicted_winner"] == 1
            else row["away_team_name"],
            "actual_winner": team_name_dict[row["winning_team"]]
            if not future_game
            else "TBD",
            "team1_logo": logo_dict[file_name_dict[row["home_team_id"]]],
            "team2_logo": logo_dict[file_name_dict[row["away_team_id"]]],
            "box": f"https://www.mlb.com/video/game/{row['game_id']}"
            if not future_game
            else f"https://www.mlb.com/news/gamepreview-{row['game_id']}",
        }

        if game["actual_winner"] == game["prediction"]:
            games_correct += 1

        processed_data.append(game)

    return render_template(
        "index.html",
        games=processed_data,
        correct_percent=int(100 * (games_correct / (games_attempted - future_games))),
        games_attempted=games_attempted,
        predictions_to_be_made=future_games,
    )


@app.route("/about")
def display_about():
    current_file = max(bucket.objects.all(), key=lambda obj: obj.key).key

    model_pickle = pickle.loads(
        s3.Bucket(os.getenv("AWS_S3_MODEL_SRC"))
        .Object(current_file)
        .get()["Body"]
        .read()
    )

    model, model_metadata = model_pickle[0]
    old_school_model, old_school_model_metadata = model_pickle[1]
    modern_model, modern_model_metadata = model_pickle[2]

    parameter_count = len(model_metadata["parameters used"].split(","))
    old_school_parameter_count = len(
        old_school_model_metadata["parameters used"].split(",")
    )
    modern_parameter_count = len(modern_model_metadata["parameters used"].split(","))
    return render_template(
        "about.html",
        date_created=model_metadata["date created"],
        model_type=model_metadata["model type"],
        parameters_used=model_metadata["parameters used"],
        parameter_count=parameter_count,
        accuracy=model_metadata["accuracy"],
        training_set_size=model_metadata["training set size"],
        testing_set_size=model_metadata["testing set size"],
        old_school_date_created=old_school_model_metadata["date created"],
        old_school_model_type=old_school_model_metadata["model type"],
        old_school_parameters_used=old_school_model_metadata["parameters used"],
        old_school_parameter_count=old_school_parameter_count,
        old_school_accuracy=old_school_model_metadata["accuracy"],
        old_school_training_set_size=old_school_model_metadata["training set size"],
        old_school_testing_set_size=old_school_model_metadata["testing set size"],
        modern_date_created=modern_model_metadata["date created"],
        modern_model_type=modern_model_metadata["model type"],
        modern_parameters_used=modern_model_metadata["parameters used"],
        modern_parameter_count=modern_parameter_count,
        modern_accuracy=modern_model_metadata["accuracy"],
        modern_training_set_size=modern_model_metadata["training set size"],
        modern_testing_set_size=modern_model_metadata["testing set size"],
    )


def predict_games(model_type):
    current_file = max(bucket.objects.all(), key=lambda obj: obj.key).key

    model_pickle = pickle.loads(
        s3.Bucket(os.getenv("AWS_S3_MODEL_SRC"))
        .Object(current_file)
        .get()["Body"]
        .read()
    )

    model, model_metadata = model_pickle[0]
    old_school_model, old_school_metadata = model_pickle[1]
    modern_model, modern_metadata = model_pickle[2]

    print("\n".join([f"{key}: {value}" for key, value in model_metadata.items()]))

    conn = create_db_connection()
    cursor = conn.cursor()

    sql = f"select * from games where predicted_winner is null"
    # sql = f"select * from games"

    cursor.execute(sql)

    column_names = [desc[0] for desc in cursor.description]

    rows = cursor.fetchall()

    df = pd.DataFrame(rows, columns=column_names)

    df = df.dropna()

    sql = f"select * from games where winning_team is null"

    cursor.execute(sql)

    rows = cursor.fetchall()

    df2 = pd.DataFrame(rows, columns=column_names)

    df = df.append(df2)

    df = df.dropna(
        subset=[
            "game_id",
            "home_team_id",
            "home_team_name",
            "away_team_id",
            "away_team_name",
            "home_pitcher",
            "home_pitcher_id",
            "home_pitcher_era",
            "home_pitcher_win_percentage",
            "home_pitcher_wins",
            "home_pitcher_losses",
            "home_pitcher_innings_pitched",
            "away_pitcher",
            "away_pitcher_id",
            "away_pitcher_era",
            "away_pitcher_win_percentage",
            "away_pitcher_wins",
            "away_pitcher_losses",
            "away_pitcher_innings_pitched",
            "home_pitcher_k_nine",
            "home_pitcher_bb_nine",
            "home_pitcher_k_bb_diff",
            "home_pitcher_whip",
            "home_pitcher_babip",
            "away_pitcher_k_nine",
            "away_pitcher_bb_nine",
            "away_pitcher_k_bb_diff",
            "away_pitcher_whip",
            "away_pitcher_babip",
        ],
        how="any",
    )

    df["away_pitcher_k_bb_ratio"] = (
        df["away_pitcher_k_nine"] / df["away_pitcher_bb_nine"]
    )
    df["home_pitcher_k_bb_ratio"] = (
        df["home_pitcher_k_nine"] / df["home_pitcher_bb_nine"]
    )

    for _, row in df.iterrows():
        pitcher_era_comp = row["away_pitcher_era"] - row["home_pitcher_era"]
        pitcher_win_percentage_comp = (
            row["away_pitcher_win_percentage"] - row["home_pitcher_win_percentage"]
        )
        pitcher_win_comp = row["away_pitcher_wins"] - row["home_pitcher_wins"]
        pitcher_losses_comp = row["away_pitcher_losses"] - row["home_pitcher_losses"]
        pitcher_innings_pitched_comp = (
            row["away_pitcher_innings_pitched"] - row["home_pitcher_innings_pitched"]
        )
        pitcher_k_nine_comp = row["away_pitcher_k_nine"] - row["home_pitcher_k_nine"]
        pitcher_bb_nine_comp = row["away_pitcher_bb_nine"] - row["home_pitcher_bb_nine"]
        pitcher_k_bb_diff_comp = (
            row["away_pitcher_k_bb_diff"] - row["home_pitcher_k_bb_diff"]
        )
        pitcher_whip_comp = row["away_pitcher_whip"] - row["home_pitcher_whip"]
        pitcher_babip_comp = row["away_pitcher_babip"] - row["home_pitcher_babip"]
        pitcher_k_bb_ratio_comp = (
            row["away_pitcher_k_bb_ratio"] - row["home_pitcher_k_bb_ratio"]
        )

        pitcher_innings_pitched_comp = (
            row["away_pitcher_innings_pitched"] - row["home_pitcher_innings_pitched"]
        )
        pitcher_k_nine_comp = row["away_pitcher_k_nine"] - row["home_pitcher_k_nine"]
        pitcher_bb_nine_comp = row["away_pitcher_bb_nine"] - row["home_pitcher_bb_nine"]
        pitcher_k_bb_diff_comp = (
            row["away_pitcher_k_bb_diff"] - row["home_pitcher_k_bb_diff"]
        )
        pitcher_whip_comp = row["away_pitcher_whip"] - row["home_pitcher_whip"]
        pitcher_babip_comp = row["away_pitcher_babip"] - row["home_pitcher_babip"]

        comparison = [
            [
                pitcher_era_comp,
                pitcher_win_percentage_comp,
                pitcher_win_comp,
                pitcher_losses_comp,
                pitcher_innings_pitched_comp,
                pitcher_k_nine_comp,
                pitcher_bb_nine_comp,
                pitcher_k_bb_diff_comp,
                pitcher_whip_comp,
                pitcher_babip_comp,
                pitcher_k_bb_ratio_comp,
            ]
        ]
        comparison = np.array(comparison)

        comparison = comparison.reshape(comparison.shape[0], -1)

        prediction = model.predict(comparison)

        sql = f"UPDATE games SET predicted_winner = {prediction[0]} WHERE game_id = {int(row['game_id'])};"

        print(sql)
        cursor.execute(sql)

    conn.commit()


if __name__ == "__main__":
    app.run(debug=True)
