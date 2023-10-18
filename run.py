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

# St_wiki_descriptionの文字数を特徴量として追加
station['St_wiki_description_length'] = station['St_wiki_description'].str.len()

# trainとtestを結合しておく
df = pd.concat([org_train, org_test], ignore_index=True)
# stationの結合
df = df.merge(station, left_on="NearestStation", right_on="Station", how="left")
# cityの結合
df = df.merge(city, on=["Prefecture", "Municipality"], how="left")

# MunicipalityごとのCoverageRatioの統計量を追加
df["MunicipalityCoverageRatio_mean"] = df.groupby("Municipality")["CoverageRatio"].transform("mean")
df["MunicipalityCoverageRatio_std"]  = df.groupby("Municipality")["CoverageRatio"].transform("std")
df["MunicipalityCoverageRatio_max"]  = df.groupby("Municipality")["CoverageRatio"].transform("max")
df["MunicipalityCoverageRatio_min"]  = df.groupby("Municipality")["CoverageRatio"].transform("min")

# MunicipalityごとのBreadthの統計量を追加
df["MunicipalityBreadth_mean"] = df.groupby("Municipality")["Breadth"].transform("mean")
df["MunicipalityBreadth_std"]  = df.groupby("Municipality")["Breadth"].transform("std")
df["MunicipalityBreadth_max"]  = df.groupby("Municipality")["Breadth"].transform("max")
df["MunicipalityBreadth_min"]  = df.groupby("Municipality")["Breadth"].transform("min")

# MunicipalityごとのTotalFloorAreaの統計量を追加
df["MunicipalityTotalFloorArea_mean"] = df.groupby("Municipality")["TotalFloorArea"].transform("mean")
df["MunicipalityTotalFloorArea_std"]  = df.groupby("Municipality")["TotalFloorArea"].transform("std")
df["MunicipalityTotalFloorArea_max"]  = df.groupby("Municipality")["TotalFloorArea"].transform("max")
df["MunicipalityTotalFloorArea_min"]  = df.groupby("Municipality")["TotalFloorArea"].transform("min")

# MunicipalityごとのFrontageの統計量を追加
df["MunicipalityFrontage_mean"] = df.groupby("Municipality")["Frontage"].transform("mean")
df["MunicipalityFrontage_std"]  = df.groupby("Municipality")["Frontage"].transform("std")
df["MunicipalityFrontage_max"]  = df.groupby("Municipality")["Frontage"].transform("max")
df["MunicipalityFrontage_min"]  = df.groupby("Municipality")["Frontage"].transform("min")

# MunicipalityごとのMaxTimeToNearestStationの統計量を追加
df["MunicipalityMaxTimeToNearestStation_mean"] = df.groupby("Municipality")["TimeToNearestStation"].transform("mean")

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

# trainの各都道府県をvalidにしてcross validation
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
    
    # valid_pred
    vl_pred = model.predict(vl_x, num_iteration=model.best_iteration)
    score = rmse(vl_y, vl_pred)
    scores.append(score)
mean_score = np.mean(scores)
print("cv", format(mean_score, ".5f"))
wandb.config["cv"] = mean_score
log_summary(model)
wandb.finish()