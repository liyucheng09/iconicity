import sys
import datasets
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from typing import List
import pandas as pd

prompt = """Iconicity is a type of sound symbolism in language. A word is iconic if it sounds like the concept or entity that it represents. On a scale from 1-7, where 1 is "not iconic at all" and 7 is "very iconic", rate the word: {word}. Output only the number and nothing else.

Your answer:"""

def load_model(model_name):
    model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True, device_map='auto')
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False, trust_remote_code=True)
    return model, tokenizer

def make_prompts(slice = None):
    df = pd.read_csv('winter.csv')
    words = df['word'].tolist()
    prompts = [prompt.format(word=word) for word in words]
    if slice:
        prompts = prompts[:slice]
    return prompts, words

if __name__ == '__main__':
    model_name, save_path, = sys.argv[1:]
    model, tokenizer = load_model(model_name)

    tokenizer.pad_token_id = model.config.eos_token_id
    generator = pipeline('text-generation', model=model, tokenizer=tokenizer, device=0)

    prompts, words = make_prompts(slice=10)
    outputs = generator(prompts, max_length=1, num_return_sequences=1)

    with open(save_path, 'w') as f:
        for word, output in zip(words, outputs):
            f.write(f'{word},{output["generated_text"]}\n')
    
    print('Done!')
