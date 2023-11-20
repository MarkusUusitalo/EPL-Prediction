import pandas as pd
from google.colab import files

uploaded = files.upload()

for fn in uploaded.keys():
  print('User uploaded file "{name}" with length {length} bytes'.format(
      name=fn, length=len(uploaded[fn])))

matches = pd.read_csv("matches.csv", index_col = 0)
matches.head()
matches.shape
matches["team"].value_counts();

matches[matches["team"] == "Liverpool"];
matches["round"].value_counts()

matches["date"] = pd.to_datetime(matches["date"])
matches["venue_code"] = matches["venue"].astype("category").cat.codes
matches["opp_code"] = matches["opponent"].astype("category").cat.codes
matches["hour"] = matches["time"].str.replace(":.+","",regex = True).astype("int")
matches["day_code"] = matches["date"].dt.dayofweek
result_mapping = {'W': 2, 'D': 1, 'L': 0}
matches["target"] = matches["result"].map(result_mapping)

from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier(n_estimators=50, min_samples_split=10, random_state=1)
train = matches[matches["date"] < '2022-01-01']
test = matches[matches["date"] > '2022-01-01']
predictors = ["venue_code", "opp_code", "hour", "day_code"]
rf.fit(train[predictors], train["target"])
preds = rf.predict(test[predictors])
from sklearn.metrics import accuracy_score
acc = accuracy_score(test["target"],preds)
acc
combined = pd.DataFrame(dict(actual=test["target"], prediction=preds))
pd.crosstab(index=combined["actual"],columns=combined["prediction"])
grouped_matches = matches.groupby("team")
def rolling_avgs(group, cols, new_cols):
  group = group.sort_values("date")
  rolling_stats = group[cols].rolling(4,closed='left').mean()
  group[new_cols] = rolling_stats
  group = group.dropna(subset=new_cols)
  return group

matches

cols = ["gf","ga","xg","xga", "pk","sh","sot","dist"]
new_cols = [f"{c}_rolling" for c in cols]
matches_rolling = matches.groupby("team").apply(lambda x: rolling_avgs(x, cols, new_cols))
matches_rolling = matches_rolling.droplevel("team")
matches_rolling.index = range(matches_rolling.shape[0])

def make_predictions(data, predictors):
  train = data[data["date"] < '2022-01-01']
  test = data[data["date"] > '2022-01-01']
  rf.fit(train[predictors], train["target"])
  preds = rf.predict(test[predictors])
  combined = pd.DataFrame(dict(actual=test["target"], predicted = preds), index=test.index)
  precision = accuracy_score(test["target"], preds)
  return combined, precision

combined, precision = make_predictions(matches_rolling, predictors + new_cols)
precision
combined
combined = combined.merge(matches_rolling[["date", "team", "opponent", "result"]], left_index = True, right_index=True)
class MissingDict(dict):
  __missing__ = lambda self, key: key

map_values = {
    "Brighton and Hove Albion": "Brighton",
    "Manchester United": "Manchester Utd",
    "Newcastle United": "Newcastle Utd",
    "Tottenham Hotspur": "Tottenham",
    "West Ham United": "West Ham",
    "Wolverhampton Wanderers": "Wolves"
}
combined.columns
mapping = MissingDict(**map_values)
combined["new_team"] = combined["team"].map(mapping)
merged = combined.merge(combined, left_on=["date","new_team"], right_on=["date","opponent"])
merged[(merged["predicted_x"] == 0) & (merged["predicted_y"] == 2)]["actual_x"].value_counts()
accuracy = 41/(41+12+15)*100
print("Accuracy: {:.2f} %".format(accuracy))
