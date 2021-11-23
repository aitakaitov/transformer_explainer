from sys import argv
from transformer_explainer.utils.explanation import Explanation
from transformer_explainer.visualisers.word_visualiser import VisualiserWordsHTML
from transformer_explainer.visualisers.sentence_visualiser import VisualiserSentencesHTML

with open(argv[1], "r", encoding='utf-8') as f:
    s = f.read()

explanation = Explanation.from_json(s)
html = VisualiserSentencesHTML.produce_html(explanation, percentile=0.95, attr_only='pos', word_processing='sum')
