import random
import argparse
import os
import pandas as pd
import torch

from bin.data.snippet import create_snippets
from bin.eval.eval_model import eval_model
from bin.eval.graph import graph_metrics
from bin.model.model import SnippetModel


def main(input_csv: str, output_csv: str, metrics_csv: str, metrics_mean_csv: str, full_data_csv: str, graph_out: str, checkpoint: str, random_seed: int, test_only: bool):

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    random.seed(2128506)
    torch.manual_seed(2128506)

    if not test_only:

        print("Processing data...")

        print("\tGetting data...")
        df = pd.read_csv(input_csv)

        print("\tCreating snippets...")
        create_snippets(df)

        print("Creating model...")
        myModel = SnippetModel(checkpoint, device)

        print("\tInserting data...")
        myModel.insert_data(df)

        print("\tTokenizing data...")
        myModel.tokenize_data()

        print("\tRunning model...")
        myModel.run_model()

        print("\tDecoding Results...")
        myModel.decode_data()

        print("\tGetting data...")
        df = myModel.get_output()

        print("\tSaving results...")
        df.to_csv(output_csv, index=False)

    print("Testing Model...")
    eval_model(output_csv, metrics_csv, full_data_csv, metrics_mean_csv)

    print("Graphing Metrics")
    graph_metrics(metrics_csv, graph_out)


if __name__ == "__main__":
    INPUT_CSV = "input/data.csv"
    OUTPUT_CSV = "output/out.csv"
    METRICS_CSV = "output/metrics.csv"
    METRICS_MEAN_CSV = "output/metrics_mean.csv"
    FULL_DATA_CSV = "output/full_out.csv"
    GRAPH_OUT = "output/metrics.png"
    CHECKPOINT = "bigcode/tiny_starcoder_py"
    RANDOM_SEED = 2128506
    TEST_ONLY = False


    parser = argparse.ArgumentParser()
    parser.add_argument('--input_csv', type=str, default=INPUT_CSV)
    parser.add_argument('--output_csv', type=str, default=OUTPUT_CSV)
    parser.add_argument('--metrics_csv', type=str, default=METRICS_CSV)
    parser.add_argument('--metrics_mean_csv', type=str, default=METRICS_MEAN_CSV)
    parser.add_argument('--full_data_csv', type=str, default=FULL_DATA_CSV)
    parser.add_argument('--graph_out', type=str, default=GRAPH_OUT)
    parser.add_argument('--checkpoint', type=str, default=CHECKPOINT)
    parser.add_argument('--random_seed', type=int, default=RANDOM_SEED)
    parser.add_argument('--test_only', type=bool, default=TEST_ONLY)


    args = parser.parse_args()

    output_dir = os.path.dirname("output/")
    os.makedirs(output_dir, exist_ok=True)

    main(args.input_csv, args.output_csv, args.metrics_csv, args.metrics_mean_csv, args.full_data_csv, args.graph_out, args.checkpoint, args.random_seed, args.test_only)
