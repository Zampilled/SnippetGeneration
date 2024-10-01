import pandas as pd

import seaborn as sns
import matplotlib.pyplot as plt


def graph_metrics(inputdir: str, outputdir: str):
    df = pd.read_csv(inputdir)




    colors = sns.color_palette()
    sns.set_theme()
    plt.figure(figsize=(12, 6))

    for idx, column in enumerate(df.columns):
        sns.kdeplot(df[column], label=column, fill=True, color=colors[idx])

    for idx, column in enumerate(df.columns):
        mean_value = df[column].mean()
        plt.axvline(x=mean_value, linestyle='--', label=f'{column} mean', alpha=0.7, color=colors[idx])

    plt.xlabel('Values')
    plt.ylabel('Density')
    plt.title('Distribution of Metrics')
    plt.legend()
    plt.savefig(outputdir)
    plt.show()
    print(df)

