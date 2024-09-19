import random


def individual_snippets(row):
    rand1 = random.randint(1, int(len(row) / 2))
    rand2 = random.randint(int(len(row) / 2), len(row) - 2)
    row = "<fim_prefix>" + row[:rand1] + "<fim_suffix>" + row[rand2:] + "<fim_middle>"
    return row


def create_snippets(df):
    df["processed"] = df["raw"].apply(individual_snippets)
    return df
