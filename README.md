# In-Context learning and text Summarization Using Large Language Models (flan-T5 LLM)


In-context learning is very inexpensive method for learning LLMs, which sometimes works for deverse set of simple applications.

In this article we are dicussiong different inference methods of LLMs.

- Without prompt engineering
- Zero-shot
- One-shot
- Few-shot
- setting model configuration

  **1. Without In-Context Learning in LLM Inference:**
  Without in-context learning, an LLM performs inference based purely on the patterns it has learned during pre-training. It does not leverage the specific context or instructions provided in the input prompt, which is typically how in-context learning enhances the model’s performance in tasks like text completion, question answering, or reasoning.

  ```python
# Setting up the evironment 
!pip install torch datasets
!pip install transformers


# Loading required modules 
from datasets import load_dataset # load_dataset class from dataset to load a data
from transformers import AutoModelForSeq2SeqLM # use for loading model LLM model 
from transformers import AutoTokenizer  # use for tokenizing in the embedding space
from transformers import GenerationConfig # use for setting the configuration of an LLM model

  
