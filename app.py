import boto3
import os
import numpy as np
import pandas as pd
import pickle
import psycopg2
import warnings

warnings.filterwarnings("ignore")


from flask import Flask, render_template
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

model, model_metadata = model_pickle


@app.route("/")
def display_games():
    predict_games()

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

    _, model_metadata = model_pickle

    parameter_count = len(model_metadata["parameters used"].split(","))
    return render_template(
        "about.html",
        date_created=model_metadata["date created"],
        model_type=model_metadata["model type"],
        parameters_used=model_metadata["parameters used"],
        parameter_count=parameter_count,
        accuracy=model_metadata["accuracy"],
        training_set_size=model_metadata["training set size"],
        testing_set_size=model_metadata["testing set size"],
    )


def predict_games():
    current_file = max(bucket.objects.all(), key=lambda obj: obj.key).key

    model_pickle = pickle.loads(
        s3.Bucket(os.getenv("AWS_S3_MODEL_SRC"))
        .Object(current_file)
        .get()["Body"]
        .read()
    )

    model, model_metadata = model_pickle

    print("\n".join([f"{key}: {value}" for key, value in model_metadata.items()]))

    conn = create_db_connection()
    cursor = conn.cursor()

    sql = f"select * from games where predicted_winner is null"

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

    for _, row in df.iterrows():
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
                pitcher_innings_pitched_comp,
                pitcher_k_nine_comp,
                pitcher_bb_nine_comp,
                pitcher_k_bb_diff_comp,
                pitcher_whip_comp,
                pitcher_babip_comp,
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
