class TokenizerHandler:
  def __init__(self, tokenizer, special_tokens=None) -> None:
    """Initialize the Tokenizer class.

    Args:
      tokenizer: A tokenizer instance to tokenize text.
      special_tokens (dict, optional): Dictionary containing special tokens.
    """
    self.tokenizer = tokenizer
    self.special_tokens = special_tokens or {}

  def eos_question(self, input_dict):
    """Adds an end token to the question in the input dictionary.

    Args:
      input_dict (dict): A dictionary containing the key 'question'.

    Returns:
      dict: The modified input dictionary.
    """
    if 'bos_token' in self.special_tokens:
      input_dict['question'] += self.special_tokens['bos_token']
    return input_dict

  def tokenize(self, input_dict):
    """Tokenizes the input dictionary containing a question.

    Args:
      input_dict (dict): A dictionary containing the key 'question'.

    Returns:
      dict: A dictionary with tokenized question.
    """
    return self.tokenizer(input_dict['question'], truncation=True)
