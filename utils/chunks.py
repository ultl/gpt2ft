import functools
import operator


class TokenizeChunker:
  def __init__(self, block_size: int) -> None:
    """Initialize the TextChunker with a specific block size.

    Parameters:
    -----------
    block_size: int
      The desired length of each tokenized block.
    """
    self.block_size = block_size

  def divide(self, tokenized_text_dict: dict) -> dict:
    """Divides the tokenized text in the examples into fixed-length blocks.

    Parameters:
    -----------
    tokenized_text_dict: dict
      A dictionary containing tokenized text as values for different keys.

    Returns:
    -----------
      dict: A dictionary with tokenized text divided into fixed-length blocks.
    """
    concatenated_examples: dict = {
      k: functools.reduce(operator.iadd, tokenized_text_dict[k], []) for k in tokenized_text_dict
    }
    # p(concatenated_examples)

    total_length = len(concatenated_examples[next(iter(tokenized_text_dict.keys()))])
    # p(total_length)

    total_length = (total_length // self.block_size) * self.block_size
    # p(total_length)

    result = {
      k: [t[i : i + self.block_size] for i in range(0, total_length, self.block_size)]
      for k, t in concatenated_examples.items()
    }

    result['labels'] = result['input_ids'].copy()
    return result
