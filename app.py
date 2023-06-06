import psycopg2
import os
import pandas as pd


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


@app.route("/")
def display_games():
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


if __name__ == "__main__":
    app.run(debug=True)
