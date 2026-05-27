## This is a test script to verify that the model can be loaded and used to compute sentence embeddings.
## from https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2

import json
import os
from transformers import AutoTokenizer, AutoModel
import torch
import torch.nn.functional as F

#Mean Pooling - Take attention mask into account for correct averaging
def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0] #First element of model_output contains all token embeddings
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)


# Sentences we want sentence embeddings for
sentences = ['This is an example sentence']

# Load model from assets
tokenizer = AutoTokenizer.from_pretrained('./assets')
model = AutoModel.from_pretrained('./assets')

# Tokenize sentences
encoded_input = tokenizer(sentences, padding=True, truncation=True, return_tensors='pt')

# Compute token embeddings
with torch.no_grad():
    model_output = model(**encoded_input)

# Perform pooling
sentence_embeddings = mean_pooling(model_output, encoded_input['attention_mask'])

# Normalize embeddings
sentence_embeddings = F.normalize(sentence_embeddings, p=2, dim=1)

# Save to assets/test_embedding.json
output_path = os.path.join('assets', 'test_embedding.json')
with open(output_path, 'w') as f:
    json.dump(sentence_embeddings[0].tolist(), f)

print(f"Sentence embeddings saved to {output_path}")
