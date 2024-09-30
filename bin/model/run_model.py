import pandas
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm

class SnippetModel:
    def __init__(self, name, device):
        self.data = None
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(name)
        self.model = AutoModelForCausalLM.from_pretrained(name).to(device)

    def insert_data(self, data):
        self.data = data.copy()

    def tokenize_one(self, row):
        return self.tokenizer.encode(row, return_tensors="pt").to(self.device)

    def tokenize_data(self):
        tqdm.pandas()
        self.data["tokenized"] = self.data["processed"].progress_apply(self.tokenize_one)

    def run_model_once(self, row):
        return self.model.generate(row, max_length=1000)

    def run_model(self):
        tqdm.pandas()
        self.data["output"] = self.data["tokenized"].progress_apply(self.run_model_once)

    def decode_one(self, row):
        return self.tokenizer.decode(row[0], skip_special_tokens=True, clean_up_tokenization_spaces=False)

    def decode_data(self):
        tqdm.pandas()
        self.data["decoded_output"] = self.data["output"].progress_apply(self.decode_one)

    def get_output(self):
        return self.data["decoded_output"]


