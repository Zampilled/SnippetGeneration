from transformers import AutoModelForCausalLM, AutoTokenizer


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
        print(self.data["processed"])
        self.data["tokenized"] = self.data["processed"].apply(self.tokenize_one)
        print(self.data)

    def run_model_once(self, row):
        return self.model.generate(row, max_length=100)

    def run_model(self):
        self.data["output"] = self.data["tokenized"].apply(self.run_model_once)

    def decode_one(self, row):
        return self.tokenizer.decode(row[0])

    def decode_data(self):
        self.data["decoded_output"] = self.data["output"].apply(self.decode_one)

    def get_output(self):
        return self.data["decoded_output"]


