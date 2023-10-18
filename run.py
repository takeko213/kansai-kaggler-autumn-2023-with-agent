import pandas as pd
import numpy as np
from sklearn.preprocessing import OrdinalEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
import wandb
from wandb.lightgbm import log_summary
import lightgbm as lgb
from config import Cfg
from utils import rmse

cfg = Cfg()
wandb.init()

org_train = pd.read_csv(cfg.input_dir + "train.csv")
org_test = pd.read_csv(cfg.input_dir + "test.csv")
station = pd.read_csv(cfg.input_dir + "station.csv")
city = pd.read_csv(cfg.input_dir + "city.csv")
sub = pd.read_csv(cfg.input_dir + "sample_submission.csv")

station.columns = ["Station", "St_Latitude", "St_Longitude", "St_wiki_description"]
city.columns = ['Prefecture', 'Municipality', 'Ci_Latitude', 'Ci_Longitude', 'Ci_wiki_description']

station["St_wiki_description"] = station["St_wiki_description"].str.lower()
city["Ci_wiki_description"] = city["Ci_wiki_description"].str.lower()

df = pd.concat([org_train, org_test], ignore_index=True)
df = df.merge(station, left_on="NearestStation", right_on="Station", how="left")
df = df.merge(city, on=["Prefecture", "Municipality"], how="left")

for agg_column in ["CoverageRatio", "Breadth", "TotalFloorArea", "Frontage", "MinTimeToNearestStation"]:
    for agg_func in ["mean", "max", "min", "std"]:
        df[f"Munic_{agg_column}_{agg_func}"] = df.groupby("Municipality")[agg_column].transform(agg_func)

for agg_column in ["FloorAreaRatio", "CoverageRatio", "BuildingYear", "TotalFloorArea"]:
    for agg_func in ["mean", "max", "min", "std"]:
        df[f"Station_{agg_column}_{agg_func}"] = df.groupby("NearestStation")[agg_column].transform(agg_func)

for agg_func in ["mean", "max", "min", "std"]:
    df[f"Station_Frontage_{agg_func}"] = df.groupby("NearestStation")["Frontage"].transform(agg_func)

for agg_func in ["mean", "max", "min", "std"]:
    df[f"Station_Area_{agg_func}"] = df.groupby("NearestStation")["Area"].transform(agg_func)

df["Municipality_CoverageRatio_rank"] = df.groupby("Municipality")["CoverageRatio"].rank()

df["NearestStation_CoverageRatio_rank"] = df.groupby("NearestStation")["CoverageRatio"].rank()

df["NearestStation_Breadth_rank"] = df.groupby("NearestStation")["Breadth"].rank()

df["NearestStation_BuildingYear_rank"] = df.groupby("NearestStation")["BuildingYear"].rank()

df["NearestStation_TotalFloorArea_rank"] = df.groupby("NearestStation")["TotalFloorArea"].rank()

df["NearestStation_Frontage_rank"] = df.groupby("NearestStation")["Frontage"].rank()

df["NearestStation_Area_rank"] = df.groupby("NearestStation")["Area"].rank()

cat_cols = [
    "Type", "Region", "FloorPlan", "LandShape", "Structure",
    "Use", "Purpose", "Direction", "Classification", "CityPlanning",
    "Renovation", "Remarks"
]

enc = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
enc.fit(org_train[cat_cols])
df[cat_cols] = enc.transform(df[cat_cols])

df["FrontageIsGreaterFlag"] = df["FrontageIsGreaterFlag"].astype(int)

df['Ci_wiki_description_word_count'] = df['Ci_wiki_description'].apply(lambda x : len(str(x).split(" ")))
tfidf = TfidfVectorizer(max_features=500)
tfidf_vec = tfidf.fit_transform(df['Ci_wiki_description'].values)
svd = TruncatedSVD(n_components=15, random_state=cfg.seed)
tfidf_svd = svd.fit_transform(tfidf_vec)

for i in range(15):
    df[f"Ci_wiki_description_svd_{i}"] = tfidf_svd[:, i]

df["Municipality_FloorAreaRatio_rank"] = df.groupby("Municipality")["FloorAreaRatio"].rank()

df['Year-BuildingYear'] = df['Year'] - df['BuildingYear']

# PrefectureのCountEncoding特徴量を追加
df["Prefecture_count"] = df["Prefecture"].map(df["Prefecture"].value_counts())

# CityPlanningのCountEncoding特徴量を追加
df["CityPlanning_count"] = df["CityPlanning"].map(df["CityPlanning"].value_counts())

target = "TradePrice"
not_use_cols = [
    "row_id", "Prefecture", "Municipality", "DistrictName", "NearestStation",
    "TimeToNearestStation", "Station", "St_wiki_description", "Ci_wiki_description",
    "MinTimeToNearestStation", target
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