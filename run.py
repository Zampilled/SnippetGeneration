from bin.data.snippet import create_snippets
from bin.data.process import get_data
from bin.model.run_model import SnippetModel

import torch


checkpoint = "bigcode/tiny_starcoder_py"
device = 'cuda' if torch.cuda.is_available() else 'cpu'

print("Getting data...")
df = get_data("input/data.csv")

print("Creating snippets...")
create_snippets(df)

print("Creating model...")
myModel = SnippetModel(checkpoint, device)

print("Inserting data...")
myModel.insert_data(df)

print("Tokenizing data...")
myModel.tokenize_data()

print("Running model...")
myModel.run_model()

print("Decoding Results...")
myModel.decode_data()

print("Getting data...")
df["output_final"] = myModel.get_output()

print("Saving results...")
df["output_final"].to_csv("output/out.csv", index=False)


