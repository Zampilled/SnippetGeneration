import random

import pandas as pd


def individual_snippets(row: str):
    """
    Creating Snippets for each row to be used in the model. Takes a random chunk of text from the middle half of the snippet and removes it.
    :param row: A string input from which snippets will be extracted and remixed with predefined markers.
    :return: A remixed string with sections from the original input separated by "<fim_prefix>", "<fim_suffix>", and "<fim_middle>".
    """
    length_row = len(row)
    rand1 = random.randint((length_row // 4) + 1, length_row // 2)
    rand2 = random.randint((length_row // 2) + 1, 3 * (length_row // 4))
    row = "<fim_prefix>" + row[:rand1] + " <fim_suffix> " + row[rand2:] + "<fim_middle>"
    return row


def create_snippets(df: pd.DataFrame):
    """
    :param df: Input DataFrame with a column "raw" containing text data to be processed.
    :return: DataFrame with an additional column "processed" containing processed text snippets.
    """
    df["processed"] = df["raw"].apply(individual_snippets)
    return df
