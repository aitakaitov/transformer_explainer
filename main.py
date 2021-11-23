from bert_1536 import Bert1536
from transformer_explainer.explainers.integrated_gradients import ExplainerIG
import tensorflow as tf
import numpy as np
from transformer_explainer.visualisers.word_visualiser import VisualiserWordsHTML
from transformer_explainer.utils.utils import tokenize

model = Bert1536()
model.load_weights("saved-weights-3")

tokenizer = Bert1536.get_tokenizer()

inf = open("input11.txt", "r", encoding='utf-8')
text = inf.read()
inf.close()

encoded = tokenize(tokenizer, text)
tokens = tokenizer.convert_ids_to_tokens(encoded.data['input_ids'])
embeddings = model.bert.bert.get_input_embeddings().weights[0]
input_embeds = tf.gather(embeddings, encoded.data['input_ids'])

explanation = ExplainerIG(model, embeddings).explain(input_embeds=np.array(input_embeds),
                                                     attention_mask=np.array(encoded.data['attention_mask']),
                                                     token_type_ids=np.array(encoded.data['token_type_ids']),
                                                     tokens=tokens, interpolation_steps=2, silent=False, x_input=True)

print(explanation.to_json())