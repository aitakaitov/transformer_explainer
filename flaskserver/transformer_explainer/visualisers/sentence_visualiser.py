from transformer_explainer.utils.explanation import Explanation
from transformer_explainer.utils.utils import get_word_list_sum, get_word_list_avg, remove_percentile_abs, scale_attributions, remove_negative
from transformer_explainer.utils.utils import remove_positive, keep_top_k_abs, get_word_html, get_html_header, get_html_footer
from transformer_explainer.utils.utils import get_html_classification_header


class VisualiserSentencesHTML:

    @staticmethod
    def produce_html(data: Explanation, percentile=0.95, word_processing='sum', sentence_processing='average',
                     attr_only='pos', sentence_limit='auto', double_sentence_limit_for_both=True):

        attributions = data.attributions
        tokens = data.tokens
        classification = data.classification

        print("[VisualiserSentencesHTML] Extracting words and attributions")
        if word_processing == 'sum':
            words, word_values = get_word_list_sum(tokens, attributions)
        elif word_processing == 'average':
            words, word_values = get_word_list_avg(tokens, attributions)

        word_values = remove_percentile_abs(word_values, percentile)

        print("[VisualiserSentencesHTML] Extracting sentences and attributions")
        if sentence_processing == 'average':
            sentences, sentence_values = VisualiserSentencesHTML.__get_sentence_list_avg(words, word_values)
        elif sentence_processing == 'sum':
            sentences, sentence_values = VisualiserSentencesHTML.__get_sentence_list_sum(words, word_values)

        if attr_only == 'pos':
            print("[VisualiserSentencesHTML] Keeping positive attributions")
            sentence_values = remove_negative(sentence_values)
        elif attr_only == 'both' or attr_only is None:
            print("[VisualiserSentencesHTML] Keeping all attributions")
            pass
        elif attr_only == 'neg':
            print("[VisualiserSentencesHTML] Keeping negative attributions")
            sentence_values = remove_positive(sentence_values)

        print("[VisualiserSentencesHTML] Scaling sentence attributions")
        sentence_values = scale_attributions(sentence_values)

        ref_limit_sentences = 8 / 77
        ref_limit_words = 8 / 359
        if sentence_limit == 'auto':
            limit = round((ref_limit_sentences * len(sentences) + ref_limit_words * len(word_values) * 0.9) / 2)
        elif sentence_limit is None:
            limit = 1000000
        else:
            limit = sentence_limit

        if sentence_processing == 'both' and double_sentence_limit_for_both:
            sentence_limit *= 2

        print("[VisualiserSentencesHTML] Limit for colored sentences is " + str(limit))

        if sentence_limit is not None:
            sentence_values = keep_top_k_abs(sentence_values, limit)

        html = "" + get_html_header()
        html += get_html_classification_header(classification)

        was_tag_start = False
        for i in range(len(sentences)):
            s_html, was_tag_start = get_word_html(sentences[i], sentence_values[i], was_tag_start)
            html += s_html

        html += get_html_footer()
        return html

    @staticmethod
    def __get_sentence_list_avg(words, word_values):
        sentences = []
        sentence_values = []
        temp_value = []
        sentence = ""
        in_www = False
        www_counter = 0
        for i in range(len(words)):
            word = words[i]
            if word in ["<", "p", "/", ">"]:
                if len(temp_value) != 0:
                    sentences.append(sentence)
                    sentence_values.append(sum(temp_value) / len(temp_value))
                sentences.append(word)
                sentence_values.append(0)
                sentence = ""
                temp_value = []
            elif word in ["www"]:
                sentence += " " + word
                in_www = True
            elif in_www:
                sentence += word
                if www_counter == 2:
                    in_www = False
                    www_counter = 0
                    sentence += " "
                if word == ".":
                    www_counter += 1
            elif word in ['.', "?", "!"]:
                sentences.append(sentence + word)
                temp_value.append(word_values[i])
                sentence_values.append(sum(temp_value) / len(temp_value))
                sentence = ""
                temp_value = []
            else:
                if word in [",", "“", "„", "%"]:
                    sentence += word
                else:
                    sentence += " " + word
                temp_value.append(word_values[i])

        return sentences, sentence_values

    @staticmethod
    def __get_sentence_list_sum(words, word_values):
        sentences = []
        sentence_values = []
        temp_value = 0
        sentence = ""
        in_www = False
        www_counter = 0
        for i in range(len(words)):
            word = words[i]
            if word in ["<", "p", "/", ">"]:
                if temp_value != 0:
                    sentences.append(sentence)
                    sentence_values.append(temp_value)
                sentences.append(word)
                sentence_values.append(0)
                sentence = ""
                temp_value = 0
            elif word in ["www"]:
                sentence += " " + word
                in_www = True
            elif in_www:
                sentence += word
                if www_counter == 2:
                    in_www = False
                    www_counter = 0
                    sentence += " "
                if word == ".":
                    www_counter += 1
            elif word in ['.', "?", "!"]:
                sentences.append(sentence + word)
                temp_value += word_values[i]
                sentence_values.append(temp_value)
                sentence = ""
                temp_value = 0
            else:
                if word in [",", "“", "„", "%"]:
                    sentence += word
                else:
                    sentence += " " + word
                temp_value += word_values[i]

        return sentences, sentence_values


