import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import (confusion_matrix,accuracy_score)
import statsmodels.api as sm
from tqdm.notebook import tqdm

data = pd.read_csv("Last5_games_data.csv", index_col=0)

data = data[data.shots_on_goal.notnull()]

#   Creating dummy variables
cat_vars=['Team','Opponent']
for var in cat_vars:
    cat_list='var'+'_'+var
    cat_list = pd.get_dummies(data[var], prefix=var)
    data1=data.join(cat_list)
    data=data1

data_vars=data.columns.values.tolist()

#   Dropping all columns that we dont need for modelling
to_keep=[i for i in data_vars if i not in cat_vars]
data_final = data[to_keep].drop(labels=["url", "Timedate", "giveaways", "takeaways", "shots_off_goal"], axis=1).fillna(0)


#   Replacing "W" with 1 and "L" with 0 for the purpose of modelling.
data_final = data_final.replace({'Bet_outcome': {"W": 1, "L": 0}})

data_final.describe()


X = data_final.loc[:, data_final.columns != 'Bet_outcome']
y = data_final.loc[:, data_final.columns == 'Bet_outcome']

#   Splitting data to train/test sets.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
columns = X_train.columns

#   Fitting model on train dataset
logit_model=sm.Logit(y_train,X_train)
result=logit_model.fit()
print(result.summary2())

yhat = result.predict(X_test)
prediction = list(map(round, yhat))

# confusion matrix
cm = confusion_matrix(y_test, prediction)
print ("Confusion Matrix : \n", cm)
 
# accuracy score of the model
print('Test accuracy = ', accuracy_score(y_test, prediction))


#   Fitting model on train dataset again, but with different package this time
logreg = LogisticRegression(solver='lbfgs', max_iter=1000)
logreg.fit(X_train, y_train.values.ravel())
y_pred = logreg.predict(X_test)

#   Predicting probabilites of 1 and 0 on test set.
y_probs = logreg.predict_proba(X_test)
final_bets = X_test.copy()
final_bets["pred"] = y_pred
final_bets["real"] = y_test
final_bets[["prob_0","prob_1"]] = y_probs

compact_final_bets = final_bets[["win_prob", "Bet", "pred", "real", "prob_0", "prob_1"]].copy()
compact_final_bets["Team"] = data["Team"]
compact_final_bets["Opponent"] = data["Opponent"]

#   Basic function for prediction based on set threshold.
def create_preds(row, threshold):
    if row["prob_1"]>threshold:
        return 1
    else:
        return 0

#   Loop for iterating over every unique team in dataset and making prediction over all probabilities threshold from 0 to 1. (Can be changed to any other column)
teams = compact_final_bets["Team"].unique()
final_table = pd.DataFrame(columns=["team","prob", "won", "profit", "return_"])

for team in tqdm(teams):

    filtered_df = compact_final_bets[compact_final_bets["Team"] == team]
    filtered_df_i = filtered_df.copy()

    for prob in np.arange(0, 1, 0.1):
        filtered_df_i[f"pred_{prob}"] = filtered_df.apply(lambda row: create_preds(row, prob), axis=1)
        filtered_df_i[f"listen_{prob}"] = filtered_df_i["Bet"]*filtered_df_i[f"pred_{prob}"]*filtered_df_i["real"]

        table_i = pd.DataFrame(columns=["team","prob", "won", "profit", "return_"], data=[["-", "-", "-","-", "-"]])
        table_i["won"] = filtered_df_i[f"listen_{prob}"].sum() 
        table_i["profit"] = table_i["won"] - filtered_df_i[f"pred_{prob}"].sum()
        table_i["prob"] = prob
        table_i["team"] = team
        table_i["return_"] = table_i["won"]/filtered_df_i[f"pred_{prob}"].sum()


        final_table = pd.concat([final_table, table_i], axis=0, ignore_index=True)

final_table = final_table.astype(
    {
        "prob": float,
        "won": float,
        "profit": float
    }
)

print(final_table)

#Filtering final_table without any NaNs in "return" column.
final_table_v1 = final_table[final_table.return_.notnull()]
