import csv
import torch
from model import NeuralNet
from data import (
    read_csv_file,
    init_team,
    update_stats,
    team_stats
)

INPUT_DIM = 5
net = NeuralNet(INPUT_DIM)
net.load_state_dict(torch.load(("laliga_model.pt"), map_location="cpu"))
net.eval()

def load_fixtures(filename):
    fixtures = []
    with open(filename, newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            fixtures.append({
                "Season": row["Season"],
                "Date": row["Date"],
                "HomeTeam": row["HomeTeam"],
                "AwayTeam": row["AwayTeam"]
            })
    return fixtures

fixtures = load_fixtures("A:\\ML\\La Liga Match Predictor\\Remaining_Fixtures.csv")

def build_fixture_features(fixture, team_stats):
    home = fixture["HomeTeam"]
    away = fixture["AwayTeam"]

    # Safety check
    if home not in team_stats or away not in team_stats:
        return None

    hs = team_stats[home]
    as_ = team_stats[away]

    # Require minimum history
    if hs["matches"] < 5 or as_["matches"] < 5:
        return None

    return [
        hs["goals_scored"] / hs["matches"],
        hs["goals_conceded"] / hs["matches"],
        as_["goals_scored"] / as_["matches"],
        as_["goals_conceded"] / as_["matches"],
        1  # home advantage
    ]

predictions = []

# Rebuild team stats using historical matches
matches = read_csv_file(
    "A:\\ML\\La Liga Match Predictor\\LaLiga_Matches.csv",
    season_start="1995-96"
)

for match in matches:
    home = match["HomeTeam"]
    away = match["AwayTeam"]

    init_team(home)
    init_team(away)

    # Update stats AFTER match, just like training
    update_stats(match)


with torch.no_grad():
    for fixture in fixtures:
        X_fix = build_fixture_features(fixture, team_stats)
        if X_fix is None:
            continue

        X_tensor = torch.tensor(X_fix, dtype=torch.float32).unsqueeze(0)

        pred = net(X_tensor)
        pred = torch.relu(pred)

        home_mu, away_mu = pred.squeeze().tolist()

        home_goals = torch.poisson(torch.tensor(home_mu)).item()
        away_goals = torch.poisson(torch.tensor(away_mu)).item()
        result = ""

        if home_goals > away_goals:
            result = "H"
        elif away_goals > home_goals:
            result = "A"
        else:
            result = "D"



        predictions.append({
            "Date": fixture["Date"],
            "HomeTeam": fixture["HomeTeam"],
            "AwayTeam": fixture["AwayTeam"],
            "PredHomeGoals": int(home_goals),
            "PredAwayGoals": int(away_goals),
            "Result": result
        })

        
def save_predictions(predictions, filename):

    with open(filename, "w", newline="", encoding="utf-8") as f:
        fieldnames = [
            "Date",
            "HomeTeam",
            "AwayTeam",
            "PredHomeGoals",
            "PredAwayGoals",
            "Result"
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(predictions)

save_predictions(predictions, "A:\\ML\\La Liga Match Predictor\\Predicted_Fixtures_Results.csv")


# print(predictions[:5])