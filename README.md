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

  
# Setting up the evironment 
```bash
!pip install torch datasets
!pip install transformers
```

# Loading required modules 
```python
from datasets import load_dataset # load_dataset class from dataset to load a data
from transformers import AutoModelForSeq2SeqLM # use for loading model LLM model 
from transformers import AutoTokenizer  # use for tokenizing in the embedding space
from transformers import GenerationConfig # use for setting the configuration of an LLM model
```

# loading dialoguesum dataset of huggingface using load_dataset class
```python
huggingface_dataset_name = 'knkarthick/dialogsum'
dataset = load_dataset(huggingface_dataset_name)
```

# Initializing tokenizer
```python
tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
```

# Loading the model
```python
model_name = 'google/flan-T5-base'
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
```

# Data Exploration 
```python
example_indices = [40,200]
dash_line = '_'.join('' for i in range(100))
print(dash_line)

for idx, index in enumerate(example_indices):
    print(dash_line)
    print('Example ', idx+1)
    print(dash_line)
    print('INPUT DIALOGUE: ')
    print(dataset['test'][index]['dialogue'])
    print(dash_line)
    print('BASELINE HUMAN SUMMARY')
    print(dataset['test'][index]['summary'])
    print(dash_line)
    print()
```

# Without any prompt engineering
```python
example_indices = [40,200]
dash_line = '_'.join('' for i in range(100))
print(dash_line)

for idx, index in enumerate(example_indices):
    dialogue = dataset['test'][index]['dialogue']
    summary = dataset['test'][index]['summary']
    inputs = tokenizer(dialogue, return_tensors='pt')
    print(inputs['input_ids'][0])
    model_output = model.generate(inputs['input_ids'], max_new_tokens=50)[0]
    print(model_output.shape)
    output = tokenizer.decode(model_output
        ,
                              skip_special_tokens = True)
    print(dash_line)
    print('Example ', idx+1)
    print(dash_line)
    # print('INPUT DIALOGUE: ')
    print(f'INPUT PROMPT: \n{dialogue}')
    print(dash_line)
    print(f'BASELINE HUMAN SUMMARY: \n{summary}')
    # print(dataset['test'][index]['summary'])
    print(dash_line)
    print(f'MODEL GENERATION - WITHOUT PROMPT ENGINEERING: \n{output}\n')
    print()
```

These are configurations that determine how much information or context is given to the model before making predictions.

**2. Zero-shot learning:**

- Description: The model is given a task or prompt without any examples of how to solve it.
- Example: Asking the LLM to classify a piece of text without showing it any labeled examples (e.g., “Is this review positive or negative?”).
- Use Case: This is useful when you want the model to generalize across tasks it hasn’t explicitly been trained for.

```python
example_indices = [40,200]

def make_prompt(index):
    dialogue = dataset['test'][index]['dialogue']
    summary = dataset['test'][index]['summary']
    prompt = f"""

    {dialogue}

    what is going on?
    """
    return prompt, dialogue, summary

for idx, index, in enumerate(example_indices):
  
  prompt, dialogue, summary = make_prompt(index)
  inputs = tokenizer(prompt, return_tensors='pt')
  model_output = model.generate(inputs['input_ids'], 
                                max_new_tokens=50)[0]
  output = tokenizer.decode(model_output,
                            skip_special_tokens=True)
  
  print(dash_line)
  print('Example ', idx+1)
  print(dash_line)
  # print('INPUT DIALOGUE: ')
  print(f'INPUT PROMPT: \n{dialogue}')
  print(dash_line)
  print(f'BASELINE HUMAN SUMMARY: \n{summary}')
  # print(dataset['test'][index]['summary'])
  print(dash_line)
  print(f'MODEL GENERATION - WITHOUT PROMPT ENGINEERING: \n{output}\n')
  print()
```

**3. One-shot learning:**

- Description: The model is given one example or instance of the task before making predictions.
- Example: Showing the model one example of a positive review and then asking it to classify the next one.
- Use Case: Effective when a single example can help the model generalize better for a given task.
```python
example_indices_full = [40]
example_indices_to_summarize = 200
def make_prompt(example_indices_full, example_index_to_summarize):
  prompt = ''
  for index in example_indices_full:
    dialogue = dataset['test'][index]['dialogue']
    summary = dataset['test'][index]['summary']
    prompt += f"""

dialogue:

{dialogue}
what is going on?
{summary}
    """
  dialogue = dataset['test'][200]['dialogue']
  prompt += f"""
  
dialogue:
{dialogue}

what is going on?
    """
  return prompt





inputs = tokenizer(one_shot_prompt, return_tensors='pt')
model_output = model.generate(inputs['input_ids'], 
                              max_new_tokens=50)[0]
output = tokenizer.decode(model_output,
                          skip_special_tokens=True)

print(dash_line)
print('Example ', idx+1)
print(dash_line)
# print('INPUT DIALOGUE: ')
print(f'INPUT PROMPT: \n{dialogue}')
print(dash_line)
print(f'BASELINE HUMAN SUMMARY: \n{summary}')
# print(dataset['test'][index]['summary'])
print(dash_line)
print(f'MODEL GENERATION - WITHOUT PROMPT ENGINEERING: \n{output}\n')
print()
```


**4. Few-shot learning:**

Description: The model is provided with a few examples of the task to infer the task’s nature.
Example: Showing the model a few examples of positive and negative reviews before asking it to classify a new review.
Use Case: The model uses these examples to better infer the patterns needed to solve the task, improving performance.

```python
example_indices_full = [40,80,120, 300,500]
example_indices_to_summarize = 200
few_shot_prompt = make_prompt(example_indices_full, example_indices_to_summarize)
print(few_shot_prompt)

inputs = tokenizer(few_shot_prompt, return_tensors='pt')
model_output = model.generate(inputs['input_ids'], 
                            generation_config=generation_config)[0]
output = tokenizer.decode(model_output,
                          skip_special_tokens=True)

print(dash_line)
print('Example ', idx+1)
print(dash_line)
# print('INPUT DIALOGUE: ')
print(f'INPUT PROMPT: \n{dialogue}')
print(dash_line)
print(f'BASELINE HUMAN SUMMARY: \n{summary}')
# print(dataset['test'][index]['summary'])
print(dash_line)
print(f'MODEL GENERATION - WITHOUT PROMPT ENGINEERING: \n{output}\n')
print()
```

**5. Generation Configuration in LLM:**
This refers to the settings that can be adjusted during the generation phase of LLM inference to control the output. Common configurations include:

- Temperature: Controls the randomness of predictions. Lower values make the output more deterministic, while higher values lead to more diverse and random outputs.
- Top-k sampling: Limits the model to choose from the top k highest probability tokens.
- Top-p (nucleus) sampling: Limits token selection to a subset where the cumulative probability is p.
- Max tokens: Limits the number of tokens in the output.
- Repetition penalty: Penalizes the model for generating repetitive sequences to ensure varied outputs.
- These configurations affect how creative, varied, or specific the generated content will be during inference.

```python
# generation_config = GenerationConfig(max_new_tokens=50)
# generation_config = GenerationConfig(max_new_tokens = 10)
# generation_config = GenerationConfig(max_new_tokens = 50, do_sample = True, temperature = 0.1)
generation_config = GenerationConfig(max_new_tokens = 50, do_sample = True, temperature = 0.5)
# generation_config = GenerationConfig(max_new_tokens = 50, do_sample = True, temperature = 1.0)
inputs = tokenizer(few_shot_prompt, return_tensors='pt')
model_output = model.generate(inputs['input_ids'], 
                            generation_config=generation_config)[0]
output = tokenizer.decode(model_output,
                          skip_special_tokens=True)

print(dash_line)
print('Example ', idx+1)
print(dash_line)
# print('INPUT DIALOGUE: ')
print(f'INPUT PROMPT: \n{dialogue}')
print(dash_line)
print(f'BASELINE HUMAN SUMMARY: \n{summary}')
# print(dataset['test'][index]['summary'])
print(dash_line)
print(f'MODEL GENERATION - WITHOUT PROMPT ENGINEERING: \n{output}\n')
print()
```



  
