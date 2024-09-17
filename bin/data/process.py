import pandas as pd

def get_data(inputdir):
    df = pd.read_csv(inputdir)
    return df
