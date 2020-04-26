import pandas as pd

# Original
from set_logger import logger

df1 = pd.read_csv('../output/sub_0.2220_LGBM+Hyper-BaseLine+count+Ridge+FeatAct.csv')
df2 = pd.read_csv('../output/sub_0.2220_LGBM+Hyper-BaseLine+count+Ridge+FeatAct+.csv')
df3 = pd.read_csv('../output/sub_0.2224_LGBM+Hyper-BaseLine+count+Ridge_SEED=15.csv')
df4 = pd.read_csv('../output/sub_0.2224_LGBM+Hyper-BaseLine+count+Ridge_SEED=30.csv')
df5 = pd.read_csv('../output/sub_0.2225_LGBM+Hyper-BaseLine+count+Ridge.csv')
df6 = pd.read_csv('../output/sub_0.2242_XGB+Hyper-BaseLine+count.csv')
df7 = pd.read_csv('../output/sub_0.2253_Cat+Hyper-BaseLine+count+Redge-TFIDF.csv')
df1['deal_probability'] = (df1['deal_probability']
			 + df2['deal_probability']
			 + df3['deal_probability']
			 + df4['deal_probability']
			 + df5['deal_probability']
			 + df6['deal_probability']
			 + df7['deal_probability'])/7

df1.to_csv("./result_tmp/sub_blend.csv", index=False)

