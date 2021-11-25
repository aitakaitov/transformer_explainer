from flask import Flask
from flask import render_template, make_response, request

from transformer_explainer.explainers.attention import ExplainerAttention
from transformer_explainer.visualisers.sentence_visualiser import VisualiserSentencesHTML
from transformer_explainer.utils.utils import tokenize

from bert_1536 import Bert1536

from tensorflow import gather
import numpy as np


app = Flask(__name__)


model = Bert1536()
model.load_weights("saved-weights-3")
explainer = ExplainerAttention(model)

embeddings = model.bert.bert.get_input_embeddings().weights[0]
tokenizer = Bert1536.get_tokenizer()


@app.route("/")
def main_page():
    return make_response(render_template("mainpage.html"))


@app.route("/presentation", methods=['POST'])
def presentation_page():
    text = request.form['input_text']
    if text == "":
        return make_response(render_template("presentation.html", text="", valid=False))

    encoded = tokenize(tokenizer, text)
    tokens = tokenizer.convert_ids_to_tokens(encoded.data['input_ids'])
    input_embeds = gather(embeddings, encoded.data['input_ids'])

    explanation = explainer.explain(input_embeds=np.array(input_embeds),
                                    attention_mask=np.array(encoded.data['attention_mask']),
                                    token_type_ids=np.array(encoded.data['token_type_ids']),
                                    tokens=tokens)

    text = VisualiserSentencesHTML.produce_html(explanation)

    return make_response(render_template("presentation.html", explanation=text, valid=True))


if __name__ == "__main__":
    app.run(debug=True)
