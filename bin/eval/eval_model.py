import pandas as pd
import sacrebleu


def em(inputs: pd.DataFrame):
    """
    Exact Match Metric
    :param inputs: DataFrame containing at least two columns, 'raw' and 'predictions'.
    :return: 1 if all values in 'raw' column are equal to corresponding values in 'predictions' column, otherwise 0.
    """
    return int(inputs["raw"] == inputs["predictions"])


def chrf(inputs: pd.DataFrame):
    """
    CHRF Metric
    :param inputs: pandas DataFrame containing 'raw' and 'predictions' columns
    :return: CHRF score calculated using sacrebleu's corpus_chrf method
    """
    return sacrebleu.corpus_chrf([inputs["raw"]], [[inputs["predictions"]]]).score


def bleu(inputs: pd.DataFrame):
    """
    BLEU Metric
    :param inputs: Pandas DataFrame containing columns 'raw' and 'predictions'.
    :return: BLEU score of the input predictions compared to the raw text.
    """
    return sacrebleu.corpus_bleu([inputs["raw"]], [[inputs["predictions"]]]).score


def ter(inputs: pd.DataFrame):
    """
    TER Metric
    :param inputs: A pandas DataFrame containing columns "raw" and "predictions".
    :return: TER (Translation Edit Rate) score as calculated by sacrebleu.corpus_ter.
    """
    return sacrebleu.corpus_ter([inputs["raw"]], [[inputs["predictions"]]]).score


def eval_model(inputdir: str, metricsdir: str, outputdir: str):
    """
    Evaluates the model using Exact Match, CHRF, BLEU, and TER. It outputs this data in full of the input data and
    as its own CSV file.
    :param inputdir: The path to the input CSV file containing the data.
    :param metricsdir: The path where the metrics results will be saved.
    :param outputdir: The path where the output CSV file with all calculations will be saved.
    :return: None
    """
    inputs = pd.read_csv(inputdir)
    print("\tRunning Exact Match...")
    inputs["em"] = inputs.apply(lambda x: em(x), axis=1)

    print("\tRunning CHRF...")
    inputs["chrf"] = inputs.apply(lambda x: chrf(x), axis=1)

    print("\tRunning BLEU...")
    inputs["bleu"] = inputs.apply(lambda x: bleu(x), axis=1)

    print("\tRunning TER...")
    inputs["ter"] = inputs.apply(lambda x: ter(x), axis=1)
    inputs.to_csv(outputdir, index=False)
    inputs = pd.DataFrame({"em": inputs["em"], "chrf": inputs["chrf"], "bleu": inputs["bleu"], "ter": inputs["ter"]})
    inputs.to_csv(metricsdir, index=False)
