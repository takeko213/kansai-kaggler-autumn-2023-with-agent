import pandas as pd
import numpy as np
from sklearn.preprocessing import OrdinalEncoder
import wandb
from wandb.lightgbm import log_summary
import lightgbm as lgb
from config import Cfg
from utils import rmse
from scipy.spatial import distance

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
df = df.merge(station, left_on="NearestStation", right_on='Station', how="left")
# cityの結合
df = df.merge(city, on=["Prefecture", "Municipality"], how="left")

# calculate euclidean distance between station and city latlong
df['euclidean_dist'] = df.apply(lambda x: 
                        distance.euclidean((x['St_Latitude'], x['St_Longitude']), 
                        (x['Ci_Latitude'], x['Ci_Longitude'])), axis=1)

# 省略

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

# 省略
