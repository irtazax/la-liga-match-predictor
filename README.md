# La Liga Match Predictor

A machine learning system that predicts La Liga match outcomes and simulates the remaining season to generate a full predicted standings table.

## Overview

This project trains a PyTorch neural network on 30 seasons of historical La Liga match data (1995–present). It engineers rolling team features, predicts scorelines using Poisson distribution sampling, and simulates all remaining fixtures to produce a final predicted league standings table with a polished visual output.

## Demo

![Predicted La Liga Standings](laliga_table.png)

## How It Works

1. **Data pipeline** — Historical match data is ingested and processed using Pandas, building rolling team statistics (average goals scored and conceded) in chronological order to prevent data leakage
2. **Model training** — A PyTorch neural network is trained on engineered features with a 70/15/15 train/validation/test split, optimized using SGD and MSE loss
3. **Prediction** — The trained model predicts expected goals for each remaining fixture; Poisson sampling is applied to generate realistic scorelines
4. **Simulation** — Remaining fixtures are simulated and aggregated into a full predicted standings table with points, wins, draws, losses, and goal difference
5. **Visualization** — Results are rendered into a styled league table using Matplotlib and PIL, complete with team logos and color-coded positions

## Tech Stack

- **Python**
- **PyTorch** — Neural network training and inference
- **Pandas** — Data ingestion and feature engineering
- **NumPy** — Numerical operations
- **Matplotlib** — League table visualization
- **PIL (Pillow)** — Image processing for team logos

## Project Structure

```
La-Liga-Match-Predictor/
├── data.py                        # Data ingestion and feature engineering
├── model.py                       # Neural network architecture and training
├── predict.py                     # Fixture prediction using trained model
├── Table.py                       # Points table calculation
├── Visual_Table.py                # Styled visual league table generation
├── laliga_model.pt                # Saved model weights
├── laliga_table.png               # Output standings table
└── README.md
```

## How To Run

### Prerequisites

```bash
pip install torch pandas numpy matplotlib pillow
```

### Steps

1. Clone the repository
```bash
git clone https://github.com/yourusername/la-liga-match-predictor.git
cd la-liga-match-predictor
```

2. Add your data files to the project directory:
   - `LaLiga_Matches.csv` — Historical match results
   - `Remaining_Fixtures.csv` — Upcoming fixtures to predict

3. Train the model
```bash
python model.py
```

4. Generate predictions
```bash
python predict.py
```

5. Visualize the standings table
```bash
python Visual_Table.py
```

## Data Format

**LaLiga_Matches.csv** expects the following columns:
| Column | Description |
|--------|-------------|
| Season | Season identifier (e.g. 1995-96) |
| Date | Match date |
| HomeTeam | Home team name |
| AwayTeam | Away team name |
| FTHG | Full time home goals |
| FTAG | Full time away goals |
| FTR | Full time result (H/A/D) |

**Remaining_Fixtures.csv** expects:
| Column | Description |
|--------|-------------|
| Season | Season identifier |
| Date | Match date |
| HomeTeam | Home team name |
| AwayTeam | Away team name |

## Model Details

- **Architecture** — 2-layer fully connected neural network (input → 100 → 2)
- **Input features** — Home avg goals scored, home avg goals conceded, away avg goals scored, away avg goals conceded, home advantage flag
- **Output** — Predicted home and away goals (continuous)
- **Loss function** — Mean Squared Error (MSE)
- **Optimizer** — SGD (lr=0.001)
- **Epochs** — 400 with batch size 128
