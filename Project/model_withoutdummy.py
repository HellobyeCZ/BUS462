import pandas as pd
import numpy as np
#from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import statsmodels.api as sm
from sklearn.metrics import (confusion_matrix,
                           accuracy_score)

data = pd.read_csv("Last5_games_data.csv", index_col=0)

data = data[data.shots_on_goal.notnull()]

data_final = data.drop(labels=['Team','Opponent', "url", "Timedate", "giveaways", "takeaways", "shots_off_goal"], axis=1).fillna(0)


data_final = data_final.replace({'Bet_outcome': {"W": 1, "L": 0}})

data_final.describe()
data_final.corr()

ax = data_final.boxplot(column=['win_prob'], by = "Bet_outcome", fontsize=10)
plt.title('')
plt.suptitle('')
ax.set_title('')
ax.set_ylabel("win_prob")
ax = plt.show()

ax = data_final.boxplot(column=['Bet'], by = "Bet_outcome")
plt.title('')
plt.suptitle('')
ax.set_title('')
ax.set_ylabel("Bet")
ax = plt.show()



X = data_final.loc[:, data_final.columns != 'Bet_outcome']
y = data_final.loc[:, data_final.columns == 'Bet_outcome']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
columns = X_train.columns

logit_model=sm.Logit(y_train,X_train)
result=logit_model.fit()
print(result.summary2())

yhat = result.predict(X_test)
prediction = list(map(round, yhat))
 
# comparing original and predicted values of y
print('Actual values', list(y_test.values))
print('Predictions :', prediction)

 
# confusion matrix
cm = confusion_matrix(y_test, prediction)
print ("Confusion Matrix : \n", cm)
 
# accuracy score of the model
print('Test accuracy = ', accuracy_score(y_test, prediction))



logreg = LogisticRegression(solver='lbfgs', max_iter=1000)
logreg.fit(X_train, y_train.values.ravel())
y_pred = logreg.predict(X_test)


y_probs = logreg.predict_proba(X_test)
final_bets = X_test.copy()
final_bets["pred"] = y_pred
final_bets["real"] = y_test
final_bets[["prob_0","prob_1"]] = y_probs

final_bets.columns

compact_final_bets = final_bets[["win_prob", "Bet", "pred", "real", "prob_0", "prob_1"]].copy()
compact_final_bets["Team"] = data["Team"]
compact_final_bets["Opponent"] = data["Opponent"]

def create_preds(row, threshold):
    if row["prob_1"]>threshold:
        return 1
    else:
        return 0


teams = compact_final_bets["Team"].unique()

final_table = pd.DataFrame(columns=["team","prob", "won", "profit"])

for team in teams:

    filtered_df = compact_final_bets[compact_final_bets["Team"] == team]
    filtered_df_i = filtered_df.copy()

    for prob in np.arange(0, 1, 0.1):
        filtered_df_i[f"pred_{prob}"] = filtered_df.apply(lambda row: create_preds(row, prob), axis=1)
        filtered_df_i[f"listen_{prob}"] = filtered_df_i["Bet"]*filtered_df_i[f"pred_{prob}"]*filtered_df_i["real"]

        table_i = pd.DataFrame(columns=["team","prob", "won", "profit"], data=[["-", "-", "-","-"]])
        table_i["won"] = filtered_df_i[f"listen_{prob}"].sum() 
        table_i["profit"] = table_i["won"] - filtered_df_i[f"pred_{prob}"].sum()
        table_i["prob"] = prob
        table_i["team"] = team


        final_table = pd.concat([final_table, table_i], axis=0, ignore_index=True)
    #print(final_table)
    #groupped_table_i = pd.DataFrame(final_table.loc[final_table['profit'].argmax()]).transpose()
    #groupped_table_i["team"] = team
    #groupped_table = pd.concat([groupped_table, groupped_table_i], ignore_index=True)

final_table = final_table.astype(
    {
        "prob": float,
        "won": float,
        "profit": float
    }
)

print(final_table)
#groupped_table.to_excel("grouped_table.xlsx")


#   1st try of automating whole process


# def see_profit_condition(data, column, condition):

#     #groupped_table = pd.DataFrame(columns=["team","prob", "won", "profit"])
#     final_table = pd.DataFrame(columns=["team","prob", "won", "profit"])


#     filtered_df = data[data[column] >= condition]
#     filtered_df_i = filtered_df.copy()

#     for prob in np.arange(0, 1, 0.1):
#         filtered_df_i[f"pred_{prob}"] = filtered_df.apply(lambda row: create_preds(row, prob), axis=1)
#         filtered_df_i[f"listen_{prob}"] = filtered_df_i["Bet"]*filtered_df_i[f"pred_{prob}"]*filtered_df_i["real"]

#         table_i = pd.DataFrame(columns=["team","prob", "won", "profit"], data=[["-", "-", "-","-"]])
#         table_i["won"] = filtered_df_i[f"listen_{prob}"].sum() 
#         table_i["profit"] = table_i["won"] - filtered_df_i.shape[0]
#         table_i["prob"] = prob
#         table_i["team"] = team


#         final_table = pd.concat([final_table, table_i], axis=0, ignore_index=True)
#     #print(final_table)
#     #groupped_table_i = pd.DataFrame(final_table.loc[final_table['profit'].argmax()]).transpose()
#     #groupped_table_i["team"] = team
#     #groupped_table = pd.concat([groupped_table, groupped_table_i], ignore_index=True)

#     final_table = final_table.astype(
#         {
#             "prob": float,
#             "won": float,
#             "profit": float
#         }
#     )

#     return final_table



#see_profit_condition(compact_final_bets, "Bet", 2)