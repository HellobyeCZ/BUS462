import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sn

data = pd.read_csv("Data_raw_202112010830.csv", index_col=0)
data["timedate"] = pd.to_datetime(data["timedate"])

data_clean = data.sort_values(by = ["timedate"])[data["bet_1"] !="-"].reset_index(drop=True)
data_clean = data_clean[data_clean["home_goals"] !="-"].reset_index(drop=True)

data_clean = data_clean.astype({"home_goals": float, "bet_1": float, "bet_0": float, "bet_2": float}, errors='raise') 

summary_stats = data_clean.describe()
summary_stats.columns = ['home_goals', 'away_goals', 'bet_1', 'bet_0', 'bet_2',
       'H_Shots_on_goal', 'A_Shots_on_goal', 'H_Goalkeeper_Saves',
       'A_Goalkeeper_Saves', 'H_Penalties', 'A_Penalties',
       'H_Penalties_in_minutes', 'A_Penalties_in_minutes',
       'H_PP_Goals', 'A_PP_Goals', 'H_SH_Goals',
       'A_SH_Goals', 'H_Bodychecks', 'A_Bodychecks', 'H_Faceoffs Won',
       'A_Faceoffs Won', 'H_Faceoffs_%', 'A_Faceoffs_%',
       'H_Empty_Net_Goals', 'A_Empty_Net_Goals', 'H_Shots_blocked',
       'A_Shots_blocked', 'H_Ztráty puku po chybách',
       'A_Ztráty puku po chybách', 'H_Zisk puku po chybách',
       'A_Zisk puku po chybách', 'H_Střely mimo branku',
       'A_Střely mimo branku']

summary_stats.to_excel("Summary_stat.xlsx")

data_clean["timedate"].min()

corr_prep = data_clean[['home_goals', 'away_goals', 'bet_1', 'bet_0', 'bet_2',
       'H_Střely na branku', 'A_Střely na branku', 'H_Zásahy brankářů',
       'A_Zásahy brankářů', 'H_Vyloučení', 'A_Vyloučení']]

corr_prep.columns = ['home_goals', 'away_goals', 'bet_1', 'bet_0', 'bet_2',
       'H_Shots_on_goal', 'A_Shots_on_goal', 'H_Goalkeeper_Saves',
       'A_Goalkeeper_Saves', 'H_Penalties', 'A_Penalties']

corrMatrix = corr_prep.corr()

sn.heatmap(corrMatrix, annot=False)
plt.show()

data_clean.describe().columns

data_clean.columns = ['timedate', 'league_round', 'home', 'away', 'home_goals', 'away_goals',
       'end_match', 'bet_1', 'bet_0', 'bet_2', 'url', 'H_shots_on_goal',
       'A_shots_on_goal', 'H_shooting_PCT', 'A_shooting_PCT',
       'H_goalie_saves', 'A_goalie_saves', 'H_saves_PCT',
       'A_saves_PCT', 'H_penalties', 'A_penalties',
       'H_PIM', 'A_PIM',
       'H_PP_goals', 'A_PP_goals', 'H_SH_goals',
       'A_SH_goals', 'H_power-play_PCT', 'A_power-play_PCT',
       'H_pen_killing_PCT', 'A_pen_killing_PCT', 'H_bodychecks',
       'A_bodychecks', 'H_faceoffs_won', 'A_faceoffs_won',
       'H_faceoffs_%', 'A_faceoffs_%', 'H_empty_net_goals',
       'A_empty_net_goals', 'H_blocked_shots', 'A_blocked_shots',
       'H_giveaways', 'A_giveaways',
       'H_takeaways', 'A_takeaways',
       'H_shots_off_goal', 'A_shots_off_goal']


data_clean = data_clean.replace({
       "end_match": {
              "Po prodloužení": "AFTER OVERTIME",
              "Konec": "FINISHED",
              "Po nájezdech": "AFTER PENALTIES"
       },
       "league_round": {
              'NHL - Play Off - Čtvrtfinále': "NHL - Play Off - Quarter-final",
              'NHL - Play Off - Finále': 'NHL - Play Off - Final',
              'NHL - Play Off - Osmifinále': 'NHL - Play Off - 1/8-Finals',
              'NHL - Play Off - Semifinále': 'NHL - Play Off - Semifinals',
              'NHL - Příprava': 'NHL - Off - Pre-season',
              'NHL - Play Off - 1/16-finále': 'NHL - Play Off - 1/16-final',
              'NHL - Skupina o umístění': 'NHL - Placement group'
       }
})

data_clean = data_clean[data_clean["end_match"]!='2. třetina']
data_clean = data_clean[data_clean.columns.drop(list(data_clean.filter(regex='_PCT')))]


data_clean["Odds_win_1"] = 1/data_clean["bet_1"]
data_clean["Odds_win_2"] = 1/data_clean["bet_2"]
data_clean["Sum_bets"] = 1/data_clean["bet_1"] + 1/data_clean["bet_0"] + 1/data_clean["bet_2"]
data_clean["H_win_prob"] = data_clean["Odds_win_1"]/data_clean["Sum_bets"]
data_clean["A_win_prob"] = data_clean["Odds_win_2"]/data_clean["Sum_bets"]


data_clean.to_csv("NHL_BUS462.csv")


