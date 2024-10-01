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

def eval_model(inputdir : str, metricsdir: str, outputdir: str):
    inputs = pd.read_csv(inputdir)
    print("\tRunning Exact Match...")
    inputs["em"] = inputs.apply(lambda x: em(x), axis=1)

    print("\tRunning CHRF...")
    inputs["chrf"] = inputs.apply(lambda x: chrf(x), axis=1)

    print("\tRunning BLEU...")
    inputs["bleu"] = inputs.apply(lambda x: bleu(x), axis=1)

    print("\tRunning TER...")
    inputs["ter"] = inputs.apply(lambda x: ter(x), axis=1)
    inputs.to_csv(outputdir)
    inputs = pd.DataFrame({"em": inputs["em"], "chrf": inputs["chrf"], "bleu": inputs["bleu"], "ter": inputs["ter"]})
    inputs.to_csv(metricsdir)