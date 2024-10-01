from bin.data.snippet import create_snippets
from bin.data.process import get_data
from bin.eval.eval_model import eval_model
from bin.eval.graph import graph_metrics
from bin.model.model import SnippetModel

import torch

checkpoint = "bigcode/tiny_starcoder_py"
device = 'cuda' if torch.cuda.is_available() else 'cpu'

print("Processing data...")

print("\tGetting data...")
df = get_data("input/data.csv")

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
df.to_csv("output/out.csv", index=False)

print("Testing Model...")
eval_model("output/out.csv", "output/metrics.csv", "output/full_data.csv")

print("Graphing Metrics")
graph_metrics("output/metrics.csv", "output/metrics.png")