from transformer_explainer.utils.explanation import Explanation
from transformer_explainer.utils.utils import get_word_list_sum, get_word_list_avg, remove_percentile_abs, scale_attributions, remove_negative
from transformer_explainer.utils.utils import remove_positive, get_word_html, get_html_header, get_html_footer
from transformer_explainer.utils.utils import get_html_classification_header


class VisualiserWordsHTML:
    @staticmethod
    def produce_html(data: Explanation, percentile=0.95, word_processing='sum', attr_only='pos'):

        attributions = data.attributions
        tokens = data.tokens
        classification = data.classification

        ref_limit_sentences = 8 / 77
        ref_limit_words = 8 / 359

        print("[VisualiserWordsHTML] Extracting words and attributions")
        if word_processing == 'sum':
            words, word_values = get_word_list_sum(tokens, attributions)
        elif word_processing == 'average':
            words, word_values = get_word_list_avg(tokens, attributions)

        word_values = remove_percentile_abs(word_values, percentile)

        print("[VisualiserWordsHTML] Scaling word attributions")
        sentence_values = scale_attributions(word_values)

        if attr_only == 'pos':
            print("[VisualiserWordsHTML] Keeping positive attributions")
            word_values = remove_negative(sentence_values)
        elif attr_only == 'both' or attr_only is None:
            print("[VisualiserWordsHTML] Keeping all attributions")
            pass
        elif attr_only == 'neg':
            print("[VisualiserWordsHTML] Keeping negative attributions")
            word_values = remove_positive(sentence_values)

        html = "" + get_html_header()
        html += get_html_classification_header(classification)

        was_tag_start = False
        for i in range(len(words)):
            s_html, was_tag_start = get_word_html(words[i], word_values[i], was_tag_start)
            html += s_html

        html += get_html_footer()
        return html
