import pandas as pd
import numpy as np
from sklearn.preprocessing import OrdinalEncoder
import wandb
from wandb.lightgbm import log_summary
import lightgbm as lgb
from config import Cfg
from utils import rmse

cfg = Cfg()
wandb.init()

# データ読み込み
org_train = pd.read_csv(cfg.input_dir + "train.csv")
org_test = pd.read_csv(cfg.input_dir + "test.csv")
station = pd.read_csv(cfg.input_dir + "station.csv")
city = pd.read_csv(cfg.input_dir + "city.csv")
sub = pd.read_csv(cfg.input_dir + "sample_submission.csv")

# 前処理
station.columns = ["Station", "St_Latitude", "St_Longitude", "St_wiki_description"]
city.columns = ['Prefecture', 'Municipality', 'Ci_Latitude', 'Ci_Longitude', 'Ci_wiki_description']

station["St_wiki_description"] = station["St_wiki_description"].str.lower()
city["Ci_wiki_description"] = city["Ci_wiki_description"].str.lower()

df = pd.concat([org_train, org_test], ignore_index=True)
df = df.merge(station, left_on="NearestStation", right_on="Station", how="left")
df = df.merge(city, on=["Prefecture", "Municipality"], how="left")

#新たな特徴量生成
df["Munic_Conv_Cvg_Ratio"] = df.groupby("Municipality")["CoverageRatio"].transform(lambda x: x.sum() / len(x))
df["Munic_Conv_Cvg_Ratio_max"] = df.groupby("Municipality")["CoverageRatio"].transform("max")
df["Munic_Conv_Cvg_Ratio_min"] = df.groupby("Municipality")["CoverageRatio"].transform("min")
df["Munic_Conv_Cvg_Ratio_std"] = df.groupby("Municipality")["CoverageRatio"].transform("std")
df["Munic_Breadth_mean"] = df.groupby("Municipality")["Breadth"].transform("mean")
df["Munic_Breadth_max"] = df.groupby("Municipality")["Breadth"].transform("max")
df["Munic_Breadth_min"] = df.groupby("Municipality")["Breadth"].transform("min")
df["Munic_Breadth_std"] = df.groupby("Municipality")["Breadth"].transform("std")

#TotalFloorAreaの特徴量追加
df["Munic_TotalFloorArea_mean"] = df.groupby("Municipality")["TotalFloorArea"].transform("mean")
df["Munic_TotalFloorArea_max"] = df.groupby("Municipality")["TotalFloorArea"].transform("max")
df["Munic_TotalFloorArea_min"] = df.groupby("Municipality")["TotalFloorArea"].transform("min")
df["Munic_TotalFloorArea_std"] = df.groupby("Municipality")["TotalFloorArea"].transform("std")

#Frontageの特徴量追加
df["Munic_Frontage_mean"] = df.groupby("Municipality")["Frontage"].transform("mean")
df["Munic_Frontage_max"] = df.groupby("Municipality")["Frontage"].transform("max")
df["Munic_Frontage_min"] = df.groupby("Municipality")["Frontage"].transform("min")
df["Munic_Frontage_std"] = df.groupby("Municipality")["Frontage"].transform("std")

# 特徴量生成
cat_cols = [
    "Type", "Region", "FloorPlan", "LandShape", "Structure",
    "Use", "Purpose", "Direction", "Classification", "CityPlanning",
    "Renovation", "Remarks"
]

# カテゴリ変数の処理(ordinal encoding)
enc = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
enc.fit(org_train[cat_cols])
df[cat_cols] = enc.transform(df[cat_cols])

# True/Falseを1/0変換
df["FrontageIsGreaterFlag"] = df["FrontageIsGreaterFlag"].astype(int)

df['Ci_wiki_description_word_count'] = df['Ci_wiki_description'].apply(lambda x : len(str(x).split(" ")))

# モデル学習
target = "TradePrice"
not_use_cols = [
    "row_id", "Prefecture", "Municipality", "DistrictName", "NearestStation",
    "TimeToNearestStation", "Station", "St_wiki_description", "Ci_wiki_description", target
]
features = [c for c in df.columns if c not in not_use_cols]

df[target] = np.log1p(df[target])
train = df[df["Prefecture"]!="Osaka Prefecture"].reset_index(drop=True)
test = df[df["Prefecture"]=="Osaka Prefecture"].reset_index(drop=True)

params = {
    'objective': 'regression',
    'boosting': 'gbdt', 
    'metric': 'rmse', 
    'learning_rate': 0.05, 
    'seed': cfg.seed
}

prefs = ['Mie Prefecture', 'Shiga Prefecture', 'Kyoto Prefecture', 'Hyogo Prefecture', 'Nara Prefecture', 'Wakayama Prefecture']
scores = []
for valid_pref in prefs:
    tr_x, tr_y = train[train["Prefecture"]!=valid_pref][features], train[train["Prefecture"]!=valid_pref][target]
    vl_x, vl_y = train[train["Prefecture"]==valid_pref][features], train[train["Prefecture"]==valid_pref][target]
    tr_data = lgb.Dataset(tr_x, label=tr_y)
    vl_data = lgb.Dataset(vl_x, label=vl_y)
    model = lgb.train(params, tr_data, valid_sets=[tr_data, vl_data], num_boost_round=20000,
                      callbacks=[
                          lgb.early_stopping(stopping_rounds=100, verbose=True), 
                          lgb.log_evaluation(100)
                          ])
    
    vl_pred = model.predict(vl_x, num_iteration=model.best_iteration)
    score = rmse(vl_y, vl_pred)
    scores.append(score)

mean_score = np.mean(scores)
print("cv", format(mean_score, ".5f"))
wandb.config["cv"] = mean_score
log_summary(model)
wandb.finish()