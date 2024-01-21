import pandas as pd
import math
import scipy.stats as st
from sklearn.preprocessing import MinMaxScaler

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.width', 500)
pd.set_option('display.expand_frame_repr', False)
pd.set_option('display.float_format', lambda x: '%.5f' % x)

df = pd.read_csv("HAFTA_4/ODEV_HAFTA4/Rating Product&SortingReviewsinAmazon/amazon_review.csv")
df.head()
df["overall"].mean()
df["reviewTime"] = pd.to_datetime(df["reviewTime"])

df["reviewTime"].max()
current_date = pd.to_datetime('2014-12-12 0:0:0')

df["day_diff"] = (current_date - df["reviewTime"]).dt.days

df.loc[df["day_diff"] <= 30, "overall"].mean()

df.loc[(df["day_diff"] > 30) & (df["day_diff"] <= 90), "overall"].mean()

df.loc[(df["day_diff"] > 90) & (df["day_diff"] <= 180), "overall"].mean()

df.loc[(df["day_diff"] > 180), "overall"].mean()

def time_based_weighted_average(dataframe, w1=28, w2=26, w3=24, w4=22):
    return dataframe.loc[df["day_diff"] <= 30, "overall"].mean() * w1 / 100 + \
           dataframe.loc[(dataframe["day_diff"] > 30) & (dataframe["day_diff"] <= 90), "overall"].mean() * w2 / 100 + \
           dataframe.loc[(dataframe["day_diff"] > 90) & (dataframe["day_diff"] <= 180), "overall"].mean() * w3 / 100 + \
           dataframe.loc[(dataframe["day_diff"] > 180), "overall"].mean() * w4 / 100

time_based_weighted_average(df)

time_based_weighted_average(df, 26, 28, 24, 22)

df.head()

df["helpful_no"] = df["total_vote"] - df["helpful_yes"]
df.head(10)
def score_pos_neg_diff(helpful_yes, helpful_no):
    return helpful_yes - helpful_no

score_pos_neg_diff(3, 0)
score_pos_neg_diff(15,6)

df["score_pos_neg_diff"] = df.apply(lambda x: score_pos_neg_diff(x["helpful_yes"], x["helpful_no"]), axis=1)
df.head(10)
def score_average_rating(helpful_yes, helpful_no):
    if helpful_yes + helpful_no == 0:
        return 0
    return helpful_yes / (helpful_yes + helpful_no)

score_average_rating(15, 45)

df["score_average_rating"] = df.apply(lambda x: score_average_rating(x["helpful_yes"], x["helpful_no"]), axis=1)
df.head(10)

# wilson_lower_bound
def wilson_lower_bound(helpful_yes, helpful_no, confidence=0.95):

    n = helpful_yes + helpful_no
    if n == 0:
        return 0
    z = st.norm.ppf(1 - (1 - confidence) / 2)
    phat = 1.0 * helpful_yes / n
    return (phat + z * z / (2 * n) - z * math.sqrt((phat * (1 - phat) + z * z / (4 * n)) / n)) / (1 + z * z / n)

df["wilson_lower_bound"] = df.apply(lambda x:wilson_lower_bound(x["helpful_yes"], x["helpful_no"]), axis=1)

df.head(10)

df.sort_values("wilson_lower_bound", ascending=False).head(20)

df.sort_values("score_pos_neg_diff", ascending=False).head(20)

df.sort_values("score_average_rating", ascending=False).head(20)
