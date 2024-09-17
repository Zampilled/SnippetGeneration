from transformers import AutoModelForCausalLM, AutoTokenizer


class SnippetModel:
    def __init__(self, name, device):
        self.raw_data = None
        self.tokenized_data = None
        self.output_data = None
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(name)
        self.model = AutoModelForCausalLM.from_pretrained(name).to(device)

    def insert_data(self, data):
        self.data = data

    def tokenize_data(self):
        self.tokenized_data = self.tokenizer.encode(self.raw_data, return_tensors="pt").to(self.device)
    def run_model(self):
        self.output_data = self.model.generate(self.tokenize_data())

    def get_output(self):
        return self.output_data


