# Instalation
In order to install and use this library to it's full capacity, you must follow these steps:
1. Install poppler with `apt-get install poppler-utils` or `brew install poppler` on mac.
2. Install tesseract with `sudo apt-get install tesseract-ocr-spa` or `brew install tesseract` on mac.

3. Install the library with the command:
```bash
pip install git+https://github.com/IsaidMosqueda/arkham.git
```

# Usage
## Finetuning
If you want to run the model fine tunning, SFTTrainer library has a pending bug, where the tokenizer
won't consider the max_lenght when creating the vector space when running the tokenizer function, 
to change that go to the file `env/lib/python3.10/site-packages/trl/trainer/utils.py`  and replace 
the line `272` from:

```python
tokenized_inputs = self.tokenizer(buffer, truncation=False)["input_ids"]
```

to:
```python
tokenized_inputs = self.tokenizer(buffer, truncation=True,max_length=self.seq_length)["input_ids"]
```

## Demo
If you want to get a hand at the user experience, take a look at the `demo.ipynb`, there you'll learn to use and manipulate the OCR
functionalities as well as use the chatbot on any file you want, with a set of models to be used.

## Research
In the research folder you may find all the steps that lead to the definition of the modules, feel free to check them. Certain files are 
to big to be added, if any file is required please feel free to get it from [here](https://drive.google.com/drive/folders/15VJ6cmEudKIdbpz6Je8InFuVsz6vsvyy?usp=sharing).

## Note
If it's the first time that you run the ´falcon´model, it's necessary that you firstly install the falcon7b-Instruct model manually, to do that run the following code after installing the library:

```python
from transformers import AutoModelForCausalLM,AutoTokenizer

model_id="tiiuae/falcon-7b-instruct"
tokenizer=AutoTokenizer.from_pretrained(model_id)
model=AutoModelForCausalLM.from_pretrained(model_id, trust_remote_code=True)
```
