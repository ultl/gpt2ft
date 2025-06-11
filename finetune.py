import math

from byeprint import p
from transformers.trainer import Trainer
from transformers.training_args import TrainingArguments

from constant import dataset, model, model_checkpoint, special_tokens, tokenizer
from utils.chunks import TokenizeChunker
from utils.tokenizer import TokenizerHandler

# p(dataset['train'][0])
# p(tokenizer)
# p(special_tokens)
pin_memory = False
max_block_length = 128
handler = TokenizerHandler(tokenizer, special_tokens=special_tokens)


post_data = dataset.remove_columns(['id', 'title', 'context', 'answers'])
# p(post_data['train']['question'][0])

post_data = post_data.map(handler.eos_question)
# p(post_data['train']['question'][0])

tokenized_dataset = post_data.map(handler.tokenize, batched=True, remove_columns=['question'])
p(tokenized_dataset)


lm_dataset = tokenized_dataset.map(
  lambda tokenized_text_dict: TokenizeChunker(block_size=max_block_length).divide(tokenized_text_dict),
  batched=True,
  batch_size=1000,
  num_proc=4,  # type: ignore
)

train_dataset = lm_dataset['train'].shuffle(seed=42).select(range(100))  # type: ignore
eval_dataset = lm_dataset['validation'].shuffle(seed=42).select(range(100))  # type: ignore


tokenizer.add_special_tokens({'pad_token': '[PAD]'})


training_args = TrainingArguments(
  f'./{model_checkpoint}-squad',
  eval_strategy='epoch',
  learning_rate=2e-5,
  weight_decay=0.01,
  push_to_hub=False,  # Change to True to push the model to the Hub
)

trainer = Trainer(
  model=model,
  args=training_args,
  train_dataset=train_dataset,
  eval_dataset=eval_dataset,
  processing_class=tokenizer,
)


eval_results = trainer.evaluate()
print(f'Perplexity: {math.exp(eval_results["eval_loss"]):.2f}')
