import csv
import pandas as pd
import matplotlib.pyplot as plt

def read_csv_file(filename):
    matches = []

    with open(filename, newline='', encoding='utf-8') as csvfile:
        reader = csv.DictReader(csvfile)

        for row in reader:

            matches.append({
                "HomeTeam": row["HomeTeam"],
                "AwayTeam": row["AwayTeam"],
                "PredHomeGoals": int(row["PredHomeGoals"]),
                "PredAwayGoals": int(row["PredAwayGoals"]),
                "Result": row["Result"]
            })

    return matches

matches = read_csv_file("A:\\ML\\La Liga Match Predictor\\Predicted_Fixtures_Results.csv")

teams = {
    "Alaves": 0,
    "Ath Bilbao": 0,
    "Ath Madrid": 0,
    "Barcelona": 0,
    "Betis": 0,
    "Celta": 0,
    "Elche": 0,
    "Espanol": 0,
    "Getafe": 0,
    "Girona": 0,
    "Levante": 0,
    "Mallorca": 0,
    "Osasuna": 0,
    "Oviedo": 0,
    "Real Madrid": 0,
    "Sociedad": 0,
    "Sevilla": 0,
    "Valencia": 0,
    "Vallecano": 0,
    "Villareal": 0
}

for m in matches:
    if m["PredHomeGoals"] > m["PredAwayGoals"]:
        teams[m["HomeTeam"]] += 3
    elif m["PredAwayGoals"] > m["PredHomeGoals"]:
        teams[m["AwayTeam"]] += 3
    else:
        teams[m["HomeTeam"]] += 1
        teams[m["AwayTeam"]] += 1



sorted_dict = dict(
sorted(teams.items(), key=lambda item: item[1], reverse=True)
)

for key, value in sorted_dict.items():
    print(key, value)

