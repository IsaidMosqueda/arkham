1. Install poppler with `apt-get install poppler-utils` or `brew install poppler` on mac.
2. Install tesseract with `sudo apt-get install tesseract-ocr-spa` or `brew install tesseract` on mac.
3. If you want to run the model fine tunning, SFTTrainer library has a pending bug, where the tokenizer
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