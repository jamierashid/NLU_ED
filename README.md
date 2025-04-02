# RoBERTa Sequence Classification Inference

This repository contains a Jupyter Notebook that runs inference using a fine-tuned RoBERTa model for sequence classification. The code processes a test dataset containing paired "Claim" and "Evidence" text, generates predictions, and saves the results to a CSV file.

---

## Table of Contents
- [Overview](#overview)
- [Requirements](#requirements)
- [Usage](#usage)
- [Attribution](#attribution)

---

## Overview

The provided code:
- Installs required packages: `transformers`, `pandas`, `torch`, and `tqdm`
- Reads a CSV file with test data containing `Claim` and `Evidence` columns
- Preprocesses the text data by converting to lowercase and stripping whitespace
- Defines a custom PyTorch `Dataset` to tokenize and prepare data for inference
- Loads a fine-tuned `RobertaForSequenceClassification` model from a specified directory
- Performs inference on the test data and saves the predictions to `test_predictions.csv`

---

## Requirements

- **Python 3.x**
- **pip** package manager

Running the demo code install requisite libraries, however they can also be installed via:
```bash
pip install transformers pandas torch tqdm
```
## Usage
#### Download the files
Download the Jupyter Notebook and model files from the following Google Drive folder:
[Google Drive Link](https://drive.google.com/drive/folders/1KFMmq3c8HZteHCkANV7PFT70h0VOgDEk?usp=drive_link)

Inside the folder is a file called: roberta-for-ed.ipynb, which is the notebook containing the code to train and demo the model. 

Inside the subdirectory named "Model" are the files required for the trained model itself:
1. model.safetensors
2. config.json

Download these 2 then update the paths as below.
#### Update Paths
In the notebook, modify the following path constants to point to the correct directories for the model and test data:

```python
MODEL_PATH = ... 
TEST_DATA_PATH = ...
```

The model path should be set to the folder containing the 2 files mentioned earlier, ```model.safetensors```and ```config.json```, and the test data path should just be set to the path to the dev dataset, wherever that is on your machine.
#### Run the Notebook

Open the notebook (.ipynb) in a Jupyter environment such as Jupyter Notebook, JupyterLab, or Google Colab.

Then run the code cell under the header "Demo Code" which will load the model and test data, perform inference, and then save the predictions to a file named test_predictions.csv.


#### Test Data:
The test CSV file (dev.csv) should include at least two columns:

Claim: The claim or statement to be evaluated.

Evidence: The supporting evidence or context for the claim.

## Attribution
#### Base Model:
This project uses the RoBERTa model from Hugging Face's Transformers library.

#### Libraries and Frameworks:

1. Transformers
2. PyTorch
3. pandas
4. tqdm

#### Data Sources
The only data source used in the creation of this model are the training and dev datasets provided in the coursework.
