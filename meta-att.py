from bert_1536 import Bert1536
from transformer_explainer.explainers import ExplainerAttention
import tensorflow as tf
import numpy as np
from transformer_explainer.utils import tokenize
from sys import argv

model = Bert1536()
model.load_weights("saved-weights-3")

tokenizer = Bert1536.get_tokenizer()

inf = open(argv[1], "r", encoding='utf-8')
text = inf.read()
inf.close()

encoded = tokenize(tokenizer, text)
tokens = tokenizer.convert_ids_to_tokens(encoded.data['input_ids'])
embeddings = model.bert.bert.get_input_embeddings().weights[0]
input_embeds = tf.gather(embeddings, encoded.data['input_ids'])

explanation = ExplainerAttention(model).explain(input_embeds=np.array(input_embeds),
                                                attention_mask=np.array(encoded.data['attention_mask']),
                                                token_type_ids=np.array(encoded.data['token_type_ids']),
                                                tokens=tokens, method='smoothgrad', steps=40)

print(explanation.to_json())