import pandas as pd
import numpy as np
from sklearn.preprocessing import OrdinalEncoder
import lightgbm as lgb
from config import Cfg
from utils import rmse

cfg = Cfg()

org_train = pd.read_csv(cfg.input_dir + "train.csv")
org_test = pd.read_csv(cfg.input_dir + "test.csv")
station = pd.read_csv(cfg.input_dir + "station.csv")
city = pd.read_csv(cfg.input_dir + "city.csv")
sub = pd.read_csv(cfg.input_dir + "sample_submission.csv")

station.columns = ["Station", "St_Latitude", "St_Longitude", "St_wiki_description"]
city.columns = ['Prefecture', 'Municipality', 'Ci_Latitude', 'Ci_Longitude', 'Ci_wiki_description']

station["St_wiki_description"] = station["St_wiki_description"].str.lower()
city["Ci_wiki_description"] = city["Ci_wiki_description"].str.lower()
station['St_wiki_description_length'] = station['St_wiki_description'].str.len()

df = pd.concat([org_train, org_test], ignore_index=True)
df = df.merge(station, left_on="NearestStation", right_on="Station", how="left")
df = df.merge(city, on=["Prefecture", "Municipality"], how="left")

statistics = ['mean', 'std', 'max', 'min']
stat_cols = ["CoverageRatio", "Breadth", "TotalFloorArea", "Frontage", "MinTimeToNearestStation", "BuildingYear", "FloorAreaRatio"]
for stat in statistics:
    for col in stat_cols:
        df[f"Municipality{col}_{stat}"] = df.groupby("Municipality")[col].transform(stat)
        df[f"NearestStation{col}_{stat}"] = df.groupby("NearestStation")[col].transform(stat)

df["NearestStation_Purpose_count"] = df.groupby("NearestStation")["Purpose"].transform("count")
df["MunicipalityTotalFloorArea_rank"] = df.groupby("Municipality")["TotalFloorArea"].rank()
df["MunicipalityFloorAreaRatio_rank"] = df.groupby("Municipality")["FloorAreaRatio"].rank()
df["NearestStationFloorAreaRatio_rank"] = df.groupby("NearestStation")["FloorAreaRatio"].rank()
df["NearestStationCoverageRatio_rank"] = df.groupby("NearestStation")["CoverageRatio"].rank()
df["NearestStationTotalFloorArea_rank"] = df.groupby("NearestStation")["TotalFloorArea"].rank()
df["NearestStationArea_rank"] = df.groupby("NearestStation")["Area"].rank()

df["YearBuildingYear_diff"] = df["Year"] - df["BuildingYear"]

df["Municipality_count"] = df.groupby("Municipality")["Municipality"].transform("count")
df["Prefecture_count"] = df.groupby("Prefecture")["Prefecture"].transform("count")
df["CityPlanning_count"] = df.groupby("CityPlanning")["CityPlanning"].transform("count")
df["Classification_count"] = df.groupby("Classification")["Classification"].transform("count")
df["Structure_count"] = df.groupby("Structure")["Structure"].transform("count")
df["Use_count"] = df.groupby("Use")["Use"].transform("count")
df["FloorPlan_count"] = df.groupby("FloorPlan")["FloorPlan"].transform("count")
df["DistinctName_count"] = df.groupby("DistrictName")["DistrictName"].transform("count")
df["LandShape_count"] = df.groupby("NearestStation")["LandShape"].transform("count")
df["NearestStation_Structure_count"] = df.groupby("NearestStation")["Structure"].transform("count")

cat_cols = [
    "Type", "Region", "FloorPlan", "LandShape", "Structure",
    "Use", "Purpose", "Direction", "Classification", "CityPlanning",
    "Renovation", "Remarks"
]

for col in cat_cols:
    df[col] = df[col].astype('category')

df["FrontageIsGreaterFlag"] = df["FrontageIsGreaterFlag"].astype(int)

for stat in ['mean', 'max', 'min', 'std']:
    df[f'NearestStation_Area_{stat}'] = df.groupby('NearestStation')['Area'].transform(stat)
    df[f'DistrictName_Area_{stat}'] = df.groupby('DistrictName')['Area'].transform(stat)
    df[f'DistrictName_BuildingYear_{stat}'] = df.groupby('DistrictName')['BuildingYear'].transform(stat)
    df[f'DistrictName_CoverageRatio_{stat}'] = df.groupby('DistrictName')['CoverageRatio'].transform(stat) # Feature addition

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
preds = np.zeros(len(test))
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
    # test
    pred = model.predict(test[features], num_iteration=model.best_iteration)
    preds += pred / 6
preds = np.expm1(preds)
sub["TradePrice"] = preds
mean_score = np.mean(scores)
print("cv", format(mean_score, ".5f"))
sub.to_csv(f"output/sub.csv", index=False)
