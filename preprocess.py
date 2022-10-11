import pandas as pd
import numpy as np

def load_data(path):
    raw_data_df = pd.read_csv(path)
    scores = raw_data_df.loc[:,['Cat' in i for i in raw_data_df.columns]].to_numpy()
    return scores