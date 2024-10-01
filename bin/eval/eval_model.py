import pandas as pd
import sacrebleu


def em(inputs : pd.DataFrame):
    return int(inputs["raw"] == inputs["predictions"])

def chrf(inputs : pd.DataFrame):
    return sacrebleu.corpus_chrf([inputs["raw"]], [[inputs["predictions"]]]).score

def bleu(inputs: pd.DataFrame):
    return sacrebleu.corpus_bleu([inputs["raw"]], [[inputs["predictions"]]]).score

def ter(inputs: pd.DataFrame):
    return sacrebleu.corpus_ter([inputs["raw"]], [[inputs["predictions"]]]).score

def eval_model(inputdir : str, outputdir: str):
    inputs = pd.read_csv(inputdir)
    print("Running Exact Match...")
    inputs["em"] = inputs.apply(lambda x: em(x), axis=1)

    print("Running CHRF...")
    inputs["chrf"] = inputs.apply(lambda x: chrf(x), axis=1)

    print("Running BLEU...")
    inputs["bleu"] = inputs.apply(lambda x: bleu(x), axis=1)

    print("Running TER...")
    inputs["ter"] = inputs.apply(lambda x: ter(x), axis=1)
    inputs.to_csv(outputdir)