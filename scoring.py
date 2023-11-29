import sys
import datasets
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline, GPT2LMHeadModel, T5ForConditionalGeneration
from typing import List
import pandas as pd
import logging
from tqdm import tqdm
from transformers.pipelines.pt_utils import KeyDataset
import re
from torch.utils.data import DataLoader
import torch
from auto_gptq import exllama_set_max_input_length

prompt = """Iconicity is a type of sound symbolism in language. A word is iconic if it sounds like the concept or entity that it represents. On a scale from 1-7, where 1 is "not iconic at all" and 7 is "very iconic", rate the word: {word}. Output only the number and nothing else.

Your answer: """

trott_prompt = """Some English words sound like what they mean. These words are iconic. You might be able to guess the meaning of such a word even if you did not know English.
    
Some words that people have rated high in iconicity are “screech,” “twirl,” and “ooze” because they sound very much like what they mean.
    
Some words that people have rated moderate in iconicity are “porcupine,” “glowing,” and “steep,” because they sound somewhat like what they mean.
    
Some words rated low in iconicity are “menu,” “amateur,” and “are,” because they do not sound at all like what they mean.
    
In this task, you are going to rate words for how iconic they are. You will rate each word on a scale from 1 to 7. A rating of 1 indicates that the word is not at all iconic and does not at all sound like what it means. 7 indicates that the word is high in iconicity and sounds very much like what it means.
    
---
    
It is important that you say the word out loud to yourself, and that you think about its meaning.
    
If you are unsure of the meaning or the pronunciation of a word, you have the option of skipping it.
    
---
    
Try to focus on the word meaning of the whole word, rather than decomposing it into parts. For example, when rating ‘butterfly’ think of the insect rather than “butter” and “fly,” and rate how well the whole meaning relates to the sound of the whole word “butterfly.”
    
---
    
On a scale from 1 (not iconic at all) to 7 (very iconic), how iconic is the word '{word}'?
    
Rating: """

trott_prompt_with_examples = """Some English words sound like what they mean. These words are iconic. You might be able to guess the meaning of such a word even if you did not know English.
    
Some words that people have rated high in iconicity are “screech: 4.8,” “twirl: 5.7,” and “ooze: 5.9” because they sound very much like what they mean.
    
Some words that people have rated moderate in iconicity are “porcupine: 3.8,” “glowing: 4.9,” and “steep: 5.3,” because they sound somewhat like what they mean.
    
Some words rated low in iconicity are “menu: 1.5,” “amateur: 2.2,” and “are: 1.3,” because they do not sound at all like what they mean.
    
In this task, you are going to rate words for how iconic they are. You will rate each word on a scale from 1 to 7. A rating of 1 indicates that the word is not at all iconic and does not at all sound like what it means. 7 indicates that the word is high in iconicity and sounds very much like what it means.
    
---
    
It is important that you say the word out loud to yourself, and that you think about its meaning.
    
If you are unsure of the meaning or the pronunciation of a word, you have the option of skipping it.
    
---
    
Try to focus on the word meaning of the whole word, rather than decomposing it into parts. For example, when rating ‘butterfly’ think of the insect rather than “butter” and “fly,” and rate how well the whole meaning relates to the sound of the whole word “butterfly.”
    
---
    
On a scale from 1 (not iconic at all) to 7 (very iconic), how iconic is the word '{word}'?
    
Rating: """

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s', filename='/mnt/fast/nobackup/users/yl02706/iconicity/scoring.log', filemode='w')

def load_model(model_name):
    if 't5' in model_name:
        model = T5ForConditionalGeneration.from_pretrained(model_name, trust_remote_code=True, device_map='auto', torch_dtype=torch.bfloat16)
    else:
        model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True, device_map='auto', use_safetensors=True)
        if '70B' in model_name:
            model = exllama_set_max_input_length(model, 4096)
    
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False, trust_remote_code=True)
    return model, tokenizer

def make_prompts(prompt, slice = None):
    df = pd.read_csv('/mnt/fast/nobackup/users/yl02706/iconicity/winter.csv')
    words = df['word'].tolist()
    prompts = [prompt.format(word=word) for word in words]
    if slice:
        prompts = prompts[:slice]
    return prompts, words

if __name__ == '__main__':
    model_name, save_path, = sys.argv[1:]
    model_real_name = model_name.split('/')[-1]
    batch_size = 8
    
    model, tokenizer = load_model(model_name)
    logging.info(f'Loaded model {model_name}')
    def tokenize_function(examples):
        return tokenizer(examples['prompts'], padding='longest')

    tokenizer.pad_token_id = model.config.eos_token_id

    types_of_prompts = {
        'trott': trott_prompt,
        'trott_with_examples': trott_prompt_with_examples
    }
    
    pad_token_id = tokenizer.pad_token_id
    decoder_input_ids = torch.tensor([[pad_token_id]], dtype=torch.long).to(model.device)

    for prompt_type, prompt in types_of_prompts.items():
        print(f'Generating {prompt_type} prompts...')
        prompts, words = make_prompts(prompt)
        ds = datasets.Dataset.from_dict({'prompts': prompts})
        ds = ds.map(tokenize_function, remove_columns=['prompts'])
        padding = lambda x: tokenizer.pad(x, padding='longest', return_tensors='pt')
        dl = DataLoader(ds, batch_size=batch_size, shuffle=False, collate_fn=padding)

        output_file = f'{save_path}/{model_real_name}_{prompt_type}.csv'
        f = open(output_file, 'w')
        f.write('word,score\n')

        candidate_labels = [str(i) for i in range(1, 8)]
        candidate_labels_ids = [tokenizer.convert_tokens_to_ids(candidate_label) for candidate_label in candidate_labels]
        candidate_labels_ids = torch.tensor(candidate_labels_ids).to(model.device)

        results = []
        for batch in tqdm(dl, desc='Generating'):
            batch = {k: v.to(model.device) for k, v in batch.items()}
            seq_lengths = batch['attention_mask'].sum(dim=-1) - 1

            if 't5' in model_name:
                outputs = model(**batch, decoder_input_ids=decoder_input_ids.repeat(batch_size, 1), return_dict=True)
            else:
                outputs = model(**batch, return_dict=True)
            probs = outputs['logits'].softmax(dim=-1)

            if 't5' in model_name:
                last_token_probs = probs[:, -1, :]
            else:
                last_token_probs = probs[torch.arange(probs.size(0)), seq_lengths]
            
            condidates_probs = last_token_probs[:, candidate_labels_ids]
            most_likely = condidates_probs.argmax(dim=-1)
            most_likely = most_likely.cpu().tolist()
            results.extend(most_likely)
        
        for word, result in zip(words, results):
            f.write(f'{word},{result}\n')

        f.close()
        print('Done!')