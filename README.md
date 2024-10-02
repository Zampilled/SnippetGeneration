# Snippet Generation

This is a test problem for the 2025 JetBrains Internships

A overview of my thought process while creating this project is available in the *THOUGHTS.md* file.

This Project uses the [Tiny Starcoder](https://huggingface.co/bigcode/tiny_starcoder_py) model to do a fill in the model task.  
I used complete python functions or classes to create a dataset for testing this model. This dataset can be seen in full at */input/data.csv*.  
This dataset is run through the model and tested on metrics (Exact Match, CHRF, BLEU, TER).   
Then the distribution of these metrics is graphed.

## Installation
First install the required dependencies with the following command
```bash
pip install -r requirements.txt
```
Then run the run.py file with the following command.
```bash
python3 run.py
```
## Results
When the code finishes running the */output* directory should have the following files:

### */output/out.csv*
The output of the data when run through the model and all steps of it including raw input, snippet generation,
tokenized input, tokenized output, and decoded output.

### */output/metrics.csv*
The metrics (EM, CHRF, BLEU, and TER) for each of the output.

### */output/full_out.csv*
 *out.csv* and *metrics.csv* combined into a full input to output and metrics csv.

### */output/metrics_mean.csv*
The mean of the metrics as a simple csv.

### */output/metrics.png*
Using a KDE plot we can plot the distribution and means of the 4 metrics and display them. Below is a sample output:

![Metric Distribution](/docs/metrics.png)