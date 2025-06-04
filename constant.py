from datasets import load_dataset
from transformers import AutoTokenizer

dataset = load_dataset('squad')
model_checkpoint = 'distilgpt2'
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint, use_fast=True)
special_tokens = tokenizer.special_tokens_map
