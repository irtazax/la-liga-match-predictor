import torch
import csv
from itertools import islice

team_stats = {}

# Read function, takes the filename, the starting season and optionally, the ending season/ the next season that is not included.
def read_csv_file(filename, season_start, season_end=None):
    matches = []

    with open(filename, newline='', encoding='utf-8') as csvfile:
        reader = csv.DictReader(csvfile)

        started = False

        for row in reader:
            season = row["Season"]

            if not started:
                if season == season_start:
                    started = True
                else:
                    continue

            if season_end is not None and season == season_end:
                break

            matches.append({
                "Season": row["Season"],
                "Date": row["Date"],
                "HomeTeam": row["HomeTeam"],
                "AwayTeam": row["AwayTeam"],
                "FTHG": int(row["FTHG"]),
                "FTAG": int(row["FTAG"]),
                "FTR": row["FTR"]
            })

    return matches

# Initializes team stats if not already present  
def init_team(team):
    if team not in team_stats:
        team_stats[team] = {
            "matches": 0,
            "goals_scored": 0,
            "goals_conceded": 0
        }

# Returns average goals scored and conceded for a team
def get_averages(team):
    stats = team_stats[team]
    if stats["matches"] == 0:
        return None  # not enough history
    return (
        stats["goals_scored"] / stats["matches"],
        stats["goals_conceded"] / stats["matches"]
    )

# Updates team stats after a match
def update_stats(match):
    home = match["HomeTeam"]
    away = match["AwayTeam"]

    hg = match["FTHG"]
    ag = match["FTAG"]

    team_stats[home]["matches"] += 1
    team_stats[away]["matches"] += 1

    team_stats[home]["goals_scored"] += hg
    team_stats[home]["goals_conceded"] += ag

    team_stats[away]["goals_scored"] += ag
    team_stats[away]["goals_conceded"] += hg

X = []
Y = []
matches = read_csv_file(r"A:\\ML\\La Liga Match Predictor\\LaLiga_Matches.csv", season_start = '1995-96')
for match in matches:  # already sorted by date
    home = match["HomeTeam"]
    away = match["AwayTeam"]

    init_team(home)
    init_team(away)

    home_avg = get_averages(home)
    away_avg = get_averages(away)

    # Skip early matches (warm-up period)
    if home_avg is None or away_avg is None:
        # Update stats but don't create training data
        update_stats(match)
        continue

    # Build X
    X.append([
        home_avg[0],  # home avg scored
        home_avg[1],  # home avg conceded
        away_avg[0],  # away avg scored
        away_avg[1],  # away avg conceded
        1  # home advantage
    ])

    # Build Y
    Y.append([match["FTHG"], match["FTAG"]])

    # NOW update stats (after using the match)
    update_stats(match)


print(X)
print(Y)


# Converting X and Y to tensors
X = torch.tensor(X, dtype=torch.float32)
Y = torch.tensor(Y, dtype=torch.float32)


# Splitting the data into training, validation and test sets
n = len(X)

train_end = int(0.7 * n)
val_end   = int(0.85 * n)

X_train = X[:train_end]
Y_train = Y[:train_end]

X_val = X[train_end:val_end]
Y_val = Y[train_end:val_end]

X_test = X[val_end:]
Y_test = Y[val_end:]

print(X_train.shape, Y_train.shape)