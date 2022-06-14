import pandas as pd
import numpy as np
from tqdm.notebook import tqdm

#   Loading raw data
data = pd.read_csv("NHL_BUS462.csv", index_col=0)

all_data = pd.DataFrame()
teams = pd.unique(data[["home","away"]].values.ravel())


#   Iterating over every row and transforming 1 match(row) to 2 rows. Once from the POV of Home team and then from POV of away team.
for team in tqdm(teams, desc = "Creating 1match2rows.csv"):
    filtered_data = data[(data["home"]==team)|(data["away"]==team)].sort_values("timedate").reset_index(drop=True)
    for index, row in tqdm(filtered_data.iterrows(), total=filtered_data.shape[0], desc = team):
        stat_i = pd.DataFrame()
        if row["home"] == team:
            stat_i = pd.DataFrame(row.filter(regex='^H_')).transpose().astype(float)
            stat_i.columns = stat_i.columns.str.strip("H_")
            stat_i["Team"] = team
            stat_i["Opponent"] = row["away"]
            stat_i["Team_Goals"] = row["home_goals"]
            stat_i["Opponent_Goals"] = row["away_goals"]
            stat_i["Bet"] = row["bet_1"]
            stat_i["Timedate"] = row["timedate"]
            
            if row["end_match"] != "FINISHED":
                stat_i["Bet_outcome"] = "L"
            elif row["home_goals"]>row["away_goals"]:
                stat_i["Bet_outcome"] = "W"
            else :
                stat_i["Bet_outcome"] = "L"
            stat_i["url"] = row["url"]

        elif row["away"] == team:
            stat_i = pd.DataFrame(row.filter(regex='^A_')).transpose().astype(float)
            stat_i.columns = stat_i.columns.str.strip("A_")
            stat_i["Team"] = team
            stat_i["Opponent"] = row["home"]
            stat_i["Team_Goals"] = row["away_goals"]
            stat_i["Opponent_Goals"] = row["home_goals"]
            stat_i["Bet"] = row["bet_2"]
            stat_i["Timedate"] = row["timedate"]
            
            if row["end_match"] != "FINISHED":
                stat_i["Bet_outcome"] = "L"
            elif row["away_goals"]>row["home_goals"]:
                stat_i["Bet_outcome"] = "W"
            else :
                stat_i["Bet_outcome"] = "L"
            stat_i["url"] = row["url"]

        all_data = pd.concat([
            all_data,
            stat_i
        ], ignore_index=True)


all_data = all_data.sort_values(by="Timedate")
all_data.to_csv("1match2rows.csv")

#   all_data = pd.read_csv("1match2rows.csv", index_col=0)


n = 5
last5_games_data = pd.DataFrame()

#   Creating with average from last n games instead of actual match statistics.
#   We dont have actual statistics from until the end of the match.
for team in tqdm(teams, desc = f"Creating Last{n}_games_data"):
    filtered_data = all_data[all_data["Team"]==team].sort_values("Timedate").reset_index(drop=True)
    for index, row in tqdm(filtered_data.iterrows(), total=filtered_data.shape[0], desc = team):
        stat_i = pd.DataFrame()
        if index<n:
            continue
        else:
            stat_i = pd.DataFrame(filtered_data.iloc[0:index].tail(n).mean(numeric_only=True)).transpose()
            stat_i["Team"] = row["Team"]
            stat_i["Opponent"] = row["Opponent"]
            stat_i["Bet"] = row["Bet"]
            stat_i["Timedate"] = row["Timedate"]
            stat_i["Bet_outcome"] = row["Bet_outcome"]
            stat_i["url"] = row["url"]

        last5_games_data = pd.concat([
                last5_games_data,
                stat_i
            ], ignore_index=True)

last5_games_data = last5_games_data.sort_values(by="Timedate")
last5_games_data.to_csv(f"Last{n}_games_data.csv")

last5_games_data = pd.read_csv(f"Last{n}_games_data.csv")