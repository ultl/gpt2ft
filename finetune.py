# source: https://github.com/omidiu/GPT-2-Fine-Tuning/blob/main/main.ipynb

from byeprint import p
from constant import dataset, tokenizer, special_tokens


p(dataset['train'][0])
p(tokenizer)
print(special_tokens)


def add_end_token_to_question(input_dict):
  input_dict['question'] += special_tokens['bos_token']
  return input_dict


dataset = dataset.remove_columns(['id', 'title', 'context', 'answers'])
dataset = dataset.map(add_end_token_to_question)
print(dataset)


def tokenize_function(input_dict):
  return tokenizer(input_dict['question'], truncation=True)


tokenized_dataset = dataset.map(tokenize_function, batched=True, num_proc=4, remove_columns=['question'])
print(tokenized_dataset)
