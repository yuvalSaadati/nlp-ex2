from transformers import RobertaTokenizer, RobertaModel
from transformers import pipeline

tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
model = RobertaModel.from_pretrained('roberta-base')
text = "I am so <mask>"
encoded_input = tokenizer(text, return_tensors='pt')
# Pass the encoded input through the RoBERTa model
output = model(**encoded_input)

# Extract the vectors for "am" and "<mask>"
# Note: In RoBERTa, sentence representations can be obtained from the last hidden states.
# "am" is at position 2 in the input, and "<mask>" is at position 5.
am_vector = output.last_hidden_state[:, 2, :]
mask_vector = output.last_hidden_state[:, 5, :]

print("Vector for 'am':", am_vector)
print("Vector for '<mask>':", mask_vector)

unmasker = pipeline('fill-mask', model='roberta-base')
print(unmasker("I am so <mask>", top_k=5))
print(unmasker("I <mask> so <mask>", top_k=5))