import pandas as pd

# Original
from set_logger import logger

df1 = pd.read_csv('../output/0.790_17_lgb_submit.csv')
df2 = pd.read_csv('../output/0.789_18_xgb_submit.csv')
# df3 = pd.read_csv('../output/0.777_19_cat_submit.csv')
df1['TARGET'] = (df1['TARGET']
	       + df2['TARGET'])/2
	#       + df3['TARGET'])/3

df1.to_csv("./result_tmp/sub_blend.csv", index=False)

