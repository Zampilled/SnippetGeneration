import random


def individual_snippets(row):
    length_row = len(row)
    rand1 = random.randint((length_row // 4) + 1, length_row // 2)
    rand2 = random.randint((length_row // 2) + 1, 3 * (length_row // 4))
    row =  "<fim_prefix>" +row[:rand1] + " <fim_suffix> " + row[rand2:] + "<fim_middle>"
    return row


def create_snippets(df):
    df["processed"] = df["raw"].apply(individual_snippets)
    return df
