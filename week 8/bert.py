# /// script
# requires-python = ">=3.13"
# dependencies = [
#     "torch",
#     "transformers",
# ]
# ///
# Requires transformers>=4.48.0
import torch.nn.functional as F
from torch import no_grad
from transformers import AutoModel, AutoTokenizer

input_texts = [
    "what is the capital of China?",
    "how to implement quick sort in python?",
    "Beijing",
    "sorting algorithms",
]

model_path = "Alibaba-NLP/gte-modernbert-base"
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModel.from_pretrained(model_path)

# Tokenize the input texts
batch_dict = tokenizer(
    input_texts, max_length=512, padding=True, truncation=True, return_tensors="pt"
)

with no_grad():
    outputs = model(**batch_dict)
embeddings = outputs.last_hidden_state[:, 0]

# (Optionally) normalize embeddings
embeddings = F.normalize(embeddings, p=2, dim=1)
scores = (embeddings[:1] @ embeddings[1:].T) * 100
scores = scores.tolist()
[print(f"{t} : {round(score)}") for score, t in zip(scores[0], input_texts[1:])]
# [[42.89073944091797, 71.30911254882812, 33.664554595947266]]


#####

input_texts = [
    "New Delhi",
    "Uzbekistan",
    "United States of America",
    "Washington DC",
]

batch_dict = tokenizer(
    input_texts, max_length=8192, padding=True, truncation=True, return_tensors="pt"
)

with no_grad():
    outputs = model(**batch_dict)

embeddings = outputs.last_hidden_state[:, 0]

# (Optionally) normalize embeddings
embeddings = F.normalize(embeddings, p=2, dim=1)
print(
    f"{input_texts[0]} to {input_texts[1]} similarity:", embeddings[0] @ embeddings[1]
)
print(
    f"{input_texts[2]} to {input_texts[3]} similarity:", embeddings[2] @ embeddings[3]
)

# relation_difference = norm(
#     (embeddings[0] - embeddings[1]) - (embeddings[2] - embeddings[3])
# ).item()
# print(f"Relation difference: {relation_difference:.4f}")


predicted_vector = embeddings[0] - embeddings[1] + embeddings[2]
predicted_vector = F.normalize(predicted_vector, p=2, dim=0)
similarity = (predicted_vector @ embeddings[3]).item()
print(
    f"{input_texts[0]} - {input_texts[1]} + {input_texts[2]} similarity to {input_texts[3]}: {100*similarity:.0f}"
)
