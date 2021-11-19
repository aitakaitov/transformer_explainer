import json
import os

import PySimpleGUI
import tensorflow as tf
from transformers import AutoTokenizer, TFAutoModel, TFBertModel
from sys import argv
import webview
import numpy as np
from alibi.explainers import IntegratedGradients


class BertClassifierForIG(tf.keras.Model):
    def __init__(self):
        super(BertClassifierForIG, self).__init__()
        self.bert = TFAutoModel.from_pretrained("UWB-AIR/Czert-B-base-cased")
        self.bert.config.attention_probs_dropout_prob = 0.1
        self.bert.config.hidden_dropout_prob = 0.1
        self.lstm = tf.keras.layers.RNN(tf.keras.layers.LSTMCell(100, dropout=0.0, recurrent_dropout=0.0))
        self.dropout_1 = tf.keras.layers.Dropout(0.0)
        self.dense_1 = tf.keras.layers.Dense(24, activation='relu', name='classifier')
        self.dropout_2 = tf.keras.layers.Dropout(0.0)
        self.dense_output = tf.keras.layers.Dense(1, activation='sigmoid', name='output', dtype='float32')

    def call(self, inputs, training=False, **kwargs):
        # parse the input arguments from the kwargs
        inputs = [kwargs['inputs_embeds'], kwargs['attention_mask'], kwargs['token_type_ids']]

        x = self.bert(None, inputs_embeds=inputs[0], attention_mask=inputs[1], token_type_ids=inputs[2], training=training).pooler_output
        x = self.dense_1(x)
        return self.dense_output(x)


# ---------------------------------------------------------------------------- NEURAL NETWORK FOR CLASSIFICATION
# vanilla implementation - good for getting gradients w.r.t. input or getting
# baseline classification results. No caveats or weird stuff here

class LongBert2(tf.keras.Model):
    def __init__(self):
        super(LongBert2, self).__init__()
        self.bert = TFAutoModel.from_pretrained("UWB-AIR/Czert-B-base-cased")
        self.bert.config.attention_probs_dropout_prob = 0.1
        self.bert.config.hidden_dropout_prob = 0.1
        self.lstm = tf.keras.layers.RNN(tf.keras.layers.LSTMCell(100, dropout=0.0, recurrent_dropout=0.0))
        self.dropout_1 = tf.keras.layers.Dropout(0.0)
        self.dense_1 = tf.keras.layers.Dense(24, activation='relu', name='classifier')
        self.dropout_2 = tf.keras.layers.Dropout(0.0)
        self.dense_output = tf.keras.layers.Dense(1, activation='sigmoid', name='output', dtype='float32')

    def call(self, inputs, training=False, **kwargs):
        x = self.split_data(inputs)

        # process each of the blocks
        # for each input (batch, 512) we get a (batch, 768) result
        results = []
        i = 0
        for block in x:
            results.append(self.bert(block, training=training).pooler_output)
            i += 1
        # concat blocks
        # we concatenate the blocks, creating a tensor (batch, 8, 768)
        concatenated = tf.stack(results, axis=1)

        x = self.lstm(concatenated)
        x = self.dropout_1(x, training=training)
        x = self.dense_1(x)
        x = self.dropout_2(x, training=training)
        return self.dense_output(x)

    def split_data(self, x) -> list:
        # split each tensor into 3 blocks of 512 tokens - we get 3 blocks with shape (batch, 512) for each input (ids, mask, tokens)
        new_ids = tf.split(x[0], [512, 512, 512], axis=-1)
        new_mask = tf.split(x[1], [512, 512, 512], axis=-1)
        new_tokens = tf.split(x[2], [512, 512, 512], axis=-1)

        # return list of tuples of (ids, mask, tokens), each 512 tokens
        out = []
        for i in range(len(new_ids)):
            out.append((new_ids[i], new_mask[i], new_tokens[i]))

        return out

    def get_config(self):
        super(LongBert2, self).get_config()


# -------------------------------------------------------------------------- NEURAL NETWORK FOR ATTENTION OUTPUT
# network, that returns attentions for a given input
# it returns attentions for the CLS tokens from the last (12th) layer, which
# are averaged over all 12 attention heads
# that means the output is of shape (1536,)
# cls_token_index has to be passed, which is the position of the CLS
# token in the 1536-long input
# since non-differentiable operations are performed, gradients cannot be computed

class LongBert2ForAttention(tf.keras.Model):
    def __init__(self):
        super(LongBert2ForAttention, self).__init__()
        self.bert = TFAutoModel.from_pretrained("UWB-AIR/Czert-B-base-cased")
        self.bert.config.attention_probs_dropout_prob = 0.1
        self.bert.config.hidden_dropout_prob = 0.1
        self.lstm = tf.keras.layers.RNN(tf.keras.layers.LSTMCell(100, dropout=0.0, recurrent_dropout=0.0))
        self.dropout_1 = tf.keras.layers.Dropout(0.0)
        self.dense_1 = tf.keras.layers.Dense(24, activation='relu', name='classifier')
        self.dropout_2 = tf.keras.layers.Dropout(0.0)
        self.dense_output = tf.keras.layers.Dense(1, activation='sigmoid', name='output', dtype='float32')

    def call(self, inputs, training=False, cls_token_index=None, **kwargs):
        # cls_token_index is the index of the CLS token, so that we can find it in it's
        # respective block

        # split data into 512 blocks
        x = self.split_data(inputs)

        # process each of the blocks
        results = []
        tokens_processed = 0
        passed_cls = False
        for block in x:
            res = self.bert(block, training=training, output_attentions=True).attentions[11]

            # we need to be aware of the CLS token, which might be in the middle of a block
            tokens_processed += 512
            # remove batch dim
            res = tf.squeeze(res)
            # average over attention heads
            res = tf.math.reduce_mean(res, axis=0)
            res = res.numpy().tolist()

            # if we haven't gotten to the CLS token or have passed it, take the last token [511],
            # which will have attention to all the preceding tokens
            if tokens_processed <= cls_token_index or passed_cls:
                res = res[511]
                results.append(res)

            # otherwise be aware of the position of the CLS token
            else:
                res = res[cls_token_index % 512]
                results.append(res)
                passed_cls = True

        # concatenate the attentions for shape (1536,)
        r = results[0] + results[1] + results[2]

        return r

    def split_data(self, x) -> list:
        new_ids = tf.split(x[0], [512, 512, 512], axis=-1)
        new_mask = tf.split(x[1], [512, 512, 512], axis=-1)
        new_tokens = tf.split(x[2], [512, 512, 512], axis=-1)

        # return list of tuples of (ids, mask, tokens), each 512 tokens
        out = []
        for i in range(len(new_ids)):
            out.append((new_ids[i], new_mask[i], new_tokens[i]))

        return out


# --------------------------------------------------------------------- NEURAL NETWORK FOR CUSTOM EMBEDDED INPUTS
# this network makes use of the inputs_embeds kwarg in TFBertModel
# to pass custom embeddings to the network, bypassing it's
# embedding layer
# this means, that all arguments have to be passed as kwargs and the
# first argument of tha call() method has to be None

class LongBert2ForIG(tf.keras.Model):
    def __init__(self):
        super(LongBert2ForIG, self).__init__()
        self.bert = TFAutoModel.from_pretrained("UWB-AIR/Czert-B-base-cased")
        self.bert.config.attention_probs_dropout_prob = 0.1
        self.bert.config.hidden_dropout_prob = 0.1
        self.lstm = tf.keras.layers.RNN(tf.keras.layers.LSTMCell(100, dropout=0.0, recurrent_dropout=0.0))
        self.dropout_1 = tf.keras.layers.Dropout(0.0)
        self.dense_1 = tf.keras.layers.Dense(24, activation='relu', name='classifier')
        self.dropout_2 = tf.keras.layers.Dropout(0.0)
        self.dense_output = tf.keras.layers.Dense(1, activation='sigmoid', name='output', dtype='float32')

    def call(self, inputs, training=False, **kwargs):
        # parse the input arguments from the kwargs
        inputs = [kwargs['inputs_embeds'], kwargs['attention_mask'], kwargs['token_type_ids']]

        # split data into 512 blocks
        x = self.split_data(inputs)
        # x = eight blocks of shape (batch, 512) for each of the three inputs

        # process each of the blocks
        # for each input (batch, 512) we get a (batch, 768) result
        results = []
        for block in x:
            res = self.bert(None, inputs_embeds=block[0], attention_mask=block[1], token_type_ids=block[2], training=training).pooler_output
            results.append(res)

        concatenated = tf.stack(results, axis=1)

        x = self.lstm(concatenated)
        x = self.dropout_1(x, training=training)
        x = self.dense_1(x)
        x = self.dropout_2(x, training=training)
        return self.dense_output(x)

    def split_data(self, x) -> list:
        new_mask = tf.split(x[1], [512, 512, 512], axis=-1)
        new_ids = tf.split(x[0], [512, 512, 512], axis=-2)
        new_tokens = tf.split(x[2], [512, 512, 512], axis=-1)

        # return list of tuples of (ids, mask, tokens), each 512 tokens
        out = []
        for i in range(len(new_ids)):
            out.append((new_ids[i], new_mask[i], new_tokens[i]))

        return out


# -------------------------------------------------------------------------------------------- WEBVIEW SENTENCES
# this webview averages the passed values of the words and colors the sentences according
# to that average
# outlier values (90th percentile) are ignored
#
# data is a dictionary, which contains results, tokenized and baseline, which are
#   results:    the data produced by the preprocessing - a list of tuples, which each contain
#               the beginning index in the tokenized sequence, the token count (n-gram length)
#               and the value for that range from the preprocessor
#               in case of sentences, there is no point in having more than one value for a
#               word, so we only take 1-grams into consideration
#
#   tokenized:  the tokenized text
#
#   baseline:   classification of the neural network

class WebViewWindowSentences:
    def __init__(self, data):
        self.was_tag_start = False
        results = data['results']
        tokenized = data['tokenized']
        baseline = data['baseline']

        # get list of words - the tokenized words can be split into more tokens
        words = self.get_word_list(tokenized)
        word_count = len(words)

        # values for each word
        word_values = [0 for _ in range(word_count)]

        # remove outlier values
        percentile = self.get_percentile(results, 0.95) # 0.95 funguje cca OK

        # get the value for each word
        last_c = results[0][1]
        word_index = 0
        for b, c, val in results:
            if c != last_c:
                word_index = 0
                last_c = c
            if abs(val) > percentile:
                continue
            #if val < 0:
            #    continue
            if c not in [1]:
                continue
            for i in range(c):
                word_values[word_index + i] = val
            word_index += 1

        # get the average value for each sentence
        sentences, vals = self.get_sentences_and_values(words, word_values)

        # perform scaling to [-1, 1]
        vals = self.scale(vals)

        # remove negative values, iow remove green
        for i in range(len(vals)):
            if vals[i] < 0:
                vals[i] = 0

        # perform percentile filtering
        #percentile = self.get_percentile(vals, 0.85)
        #non_zero_count = 0
        #for i in range(len(vals)):
        #    if vals[i] < percentile:
        #        vals[i] = 0
        #    else:
        #        non_zero_count += 1
        #print(str(non_zero_count))
        print("words: " + str(len(word_values)))
        print("sentences: " + str(len(vals)))
        default_perc = 0.85
        non_zero_count = 10000

        ref_limit_sentences = 8 / 77    # vypadá to, že je třeba kompenzovatpro počet vět a slov
        ref_limit_words = 8 / 359
        print("limit sentences: " + str(ref_limit_sentences * len(sentences)))
        print("limit words: " + str(ref_limit_words * len(word_values)))
        print("actual limit: " + str((ref_limit_sentences * len(sentences) + ref_limit_words * len(word_values)) / 2))
        #ref_limit_words = ref_limit_sentences
        limit = round((ref_limit_sentences * len(sentences) + ref_limit_words * len(word_values)) / 2)
        print("limit: " + str(limit))


        while(non_zero_count > limit):
            percentile = self.get_percentile(vals, default_perc)
            non_zero_count = 0
            for i in range(len(vals)):
                if vals[i] < percentile:
                    vals[i] = 0
                else:
                    non_zero_count += 1
            default_perc += 0.02
            print(str(non_zero_count))
        print(str(non_zero_count))


        # if we have too much red, tone it down and keep the most significant K values
        #if non_zero_count > 8:
        #    for i in range(non_zero_count - 8):
        #        _min = min(vals)
        #        idx = vals.index(_min)
        #        vals[idx] = 0
        #
        #    for i in range(len(vals)):
        #        vals[i] = 0

        # display
        classification = ""
        if baseline < 0.5:
            classification = "Nekomerční článek"
        elif baseline >= 0.5:
            classification = "Komerční článek"

        html = '<html lang="cs"><head><meta http-equiv="Content-Type" content="text/html; charset=UTF-8"></head><body style="font-size: 28px;font-family:Arial"><strong>'
        html += '<div style="width:100%;text-align:center;font-size: 25px;padding-bottom:10px">Predikce sítě: {:0.4f}'.format(
            baseline) + ' -> ' + classification + '</div>'
        temp = ""
        for i in range(len(vals)):
            html += self.get_word_html(sentences[i], vals[i])
            temp += self.get_word_html(sentences[i], vals[i])

        html += "</strong></body></html>"

        print(temp)

        webview.create_window("Vizualizace", html=html)
        webview.start()

    def get_sentences_and_values(self, words, word_values):
        """
        Returns sentences and their respective averaged values
        based on the passed words and word values
        :param words:
        :param word_values:
        :return:
        """
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

    def get_percentile(self, results, percentile):
        """
        Remove outlier values above specified percentile
        :param results:
        :param percentile:
        :return:
        """
        vals = []
        if isinstance(results[0], float) or isinstance(results[0], int):
            for v in results:
                vals.append(abs(v))
        else:
            for b, c, v in results:
                vals.append(abs(v))
        vals.sort()
        return vals[int(percentile * (len(vals) - 1))]

    def get_word_html(self, word, val):
        """
        Given string and its value, get HTML representation of that string
        :param word:
        :param val:
        :return:
        """
        # we dont color the paragraphs
        if word == "<":
            # word = "&lt"
            self.was_tag_start = True
            return "<"
        elif word == ">":
            # word = "&gt"
            self.was_tag_start = False
            return ">"
        if self.was_tag_start:
            return word
        if val < 0:
            #html = '<span style="background-color:rgb(' + str(int(255 + val * 255)) + ', ' + str(255) + ', ' + str(
            #    int(255 + val * 255)) + ');">'
            html = '<span>'#'<span style="color:rgb(128, 128, 128);">'
        else:
            #html = '<span style="background-color:rgb(' + str(255) + ", " + str(int(255 - val * 255)) + "," + str(
            #    int(255 - val * 255)) + ');">'
            html = '<span style="color:rgb(' + str(int(128 * val + 127)) + ", " + str(128 - int(64 * val)) + "," + str(
                128 - int(64 * val)) + ');">'
            #html = '<span style="color:rgb(' + str(int(128 * val + 127)) + ", " + str(128 - int(64 * val)) + "," + str(
            #    128 - int(64 * val)) + ');">'
        if word not in ["!", "?", ".", ",", '"']:
            html = " " + html + word
        else:
            html = html + word
        html += "</span>"
        return html

    def scale(self, vals):
        """
        Scale the values to [-1, 1]
        :param vals:
        :return:
        """
        _max = -1
        for val in vals:
            if abs(val) > _max:
                _max = abs(val)

        for i in range(len(vals)):
            vals[i] = vals[i] / _max

        return vals

    def get_word_list(self, tokens):
        """
        Get list of words given tokenized text
        :param tokens:
        :return:
        """
        words = []
        word = tokens[0]
        for i in range(1, min(len(tokens), 1534)):
            if "##" in tokens[i]:
                word += tokens[i][2:]
            else:
                words.append(word)
                word = tokens[i]
        words.append(word)
        return words


# ------------------------------------------------------------------------------------------------ WEBVIEW WORDS
# this webview averages the passed values of the words and colors the words according
# to that average
# outlier values (90th percentile) are ignored
#
# data is a dictionary, which contains results, tokenized and baseline, which are
#   results:    the data produced by the preprocessing - a list of tuples, which each contain
#               the beginning index in the tokenized sequence, the token count (n-gram length)
#               and the value for that range from the preprocessor
#               in case of words, we can make a case for in having more than one value for a
#               word, since larger n-grams can represent the value of the word in context,
#               so we take that into account
#
#   tokenized:  the tokenized text
#
#   baseline:   classification of the neural network

class WebViewWindowWords:
    def __init__(self, data):
        self.was_tag_start = False
        self.ns = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
        results = data['results']
        tokenized = data['tokenized']
        baseline = data['baseline']

        # get list of words
        words = self.get_word_list(tokenized)
        word_count = len(words)

        # create array for the values
        word_values = [[[] for _ in self.ns] for _ in range(word_count)]

        # remove outlier values
        percentile = self.get_percentile(results, 0.95)

        # for each tuple in results, see the beginning index of the n-gram, the size of the n-gram
        # and the resulting value and add it to word_values
        last_c = results[0][1]
        word_index = 0
        for b, c, val in results:
            if c != last_c:
                word_index = 0
                last_c = c
            if abs(val) > percentile:
                continue
            #if val < 0:
            #    continue
            if c not in [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]:
                continue
            for i in range(c):
                word_values[word_index + i][self.ns.index(c)].append(val)
            word_index += 1

        # now we average the recorded values for each word
        vals = self.use_average(word_values)
        # and scale the results to [-1, 1]
        vals = self.scale(vals)

        # display
        classification = ""
        if baseline < 0.5:
            classification = "Nekomerční článek"
        elif baseline >= 0.5:
            classification = "Komerční článek"

        html = '<html lang="cs"><head><meta http-equiv="Content-Type" content="text/html; charset=UTF-8"></head><body style="font-size: 20px;font-family=arial">'
        html += '<div style="width:100%;text-align:center;font-size: 25px;padding-bottom:10px">Predikce sítě: {:0.4f}'.format(
            baseline) + ' -> ' + classification + '</div>'

        output = ""
        for i in range(len(vals)):
            html += self.get_word_html(words[i], vals[i])
            output += self.get_word_html(words[i], vals[i])

        print(output)

        html += "</body></html>"

        webview.create_window("Vizualizace", html=html)
        webview.start()

    def use_average(self, w_vals):
        """
        Given word values for each word, average the out
        :param w_vals:
        :return:
        """
        res = [0 for _ in range(len(w_vals))]
        # we need to go trough each word
        for i in range(len(w_vals)):
            _sum = 0
            _count = 0
            # for each word, we have i possible n-gram sizes
            for j in range(len(w_vals[i])):
                # for each size, sum the values (one word can be in multiple n-grams of the same size)
                for k in range(len(w_vals[i][j])):
                    _sum += w_vals[i][j][k]
                    _count += 1
            # try to make an average - for some words, they can be in no n-grams, so we check for that
            try:
                res[i] = _sum / _count
            except ZeroDivisionError:
                res[i] = 0
        return res

    def get_percentile(self, results, percentile):
        """
        Get specified percentile absolute value
        :param results:
        :param percentile:
        :return:
        """
        vals = []
        for b, c, v in results:
            vals.append(abs(v))

        vals.sort()
        return vals[int(percentile * (len(vals) - 1))]

    def use_max(self, w_vals):
        """
        Alternative to averaging - we can use maximum value for each word
        The structure is more or less identical to the use_avg()
        :param w_vals:
        :return:
        """
        temp = [[] for _ in range(len(w_vals))]
        for i in range(len(w_vals)):
            for j in range(len(w_vals[i])):
                _max = 0
                real_value = 0
                for k in range(len(w_vals[i][j])):
                    if abs(w_vals[i][j][k]) > _max:
                        _max = abs(w_vals[i][j][k])
                        real_value = w_vals[i][j][k]
                temp[i].append(real_value)

        output = [0 for _ in range(len(w_vals))]
        for i in range(len(temp)):
            _max = 0
            real_val = 0
            for j in range(len(temp[i])):
                if abs(temp[i][j]) > _max:
                    _max = abs(temp[i][j])
                    real_val = temp[i][j]
            output[i] = real_val

        return output

    def get_word_html(self, word, val):
        """
        Given a string and it's value, return its HTML representation
        :param word:
        :param val:
        :return:
        """
        # we don't color the paragraphs
        if word == "<":
            # word = "&lt"
            self.was_tag_start = True
            return "<"
        elif word == ">":
            # word = "&gt"
            self.was_tag_start = False
            return ">"
        if self.was_tag_start:
            return word
        if val < 0:
            #html = '<span style="background-color:rgb(' + str(int(255 + val * 255)) + ', ' + str(255) + ', ' + str(
            #    int(255 + val * 255)) + ');">'
            html = "<span>"
        else:
            html = '<span style="background-color:rgb(' + str(255) + ", " + str(int(255 - val * 255)) + "," + str(
                int(255 - val * 255)) + ');">'
        if word not in ["!", "?", ".", ",", '"']:
            html = " " + html + word
        else:
            html = html + word
        html += "</span>"
        return html

    def scale(self, vals):
        """
        scale the values to [-1, 1]
        :param vals:
        :return:
        """
        _max = -1
        for val in vals:
            if abs(val) > _max:
                _max = abs(val)

        for i in range(len(vals)):
            temp = vals[i]
            vals[i] = vals[i] / _max
            if (temp > 0 and vals[i] < 0) or (temp < 0 and vals[i] > 0):
                print("Scaling error")
        return vals

    def get_word_list(self, tokens):
        """
        Given tokenized text, returns a list of words
        :param tokens:
        :return:
        """
        words = []
        word = tokens[0]
        for i in range(1, min(len(tokens), 1534)):
            if "##" in tokens[i]:
                word += tokens[i][2:]
            else:
                words.append(word)
                word = tokens[i]
        words.append(word)
        return words


class PreprocessorSmoothGRADxInputs:
    def __init__(self):
        model, tokenizer = self.get_model(LongBert2)
        self.tokenizer = tokenizer
        self.model = model

        # we need to get the embeddings to perform the interpolation
        self.embeddings = self.model.bert.bert.get_input_embeddings().weights[0]

    def get_model(self, model_class):
        model = model_class()
        model.load_weights("saved-weights-1")
        return model, AutoTokenizer.from_pretrained("UWB-AIR/Czert-B-base-cased")

    def classify_example(self, _input, model):
        el = (
            tf.expand_dims(tf.convert_to_tensor(_input[0]), axis=0),
            tf.expand_dims(tf.convert_to_tensor(_input[1]), axis=0),
            tf.expand_dims(tf.convert_to_tensor(_input[2]), axis=0)
        )
        res = model(el, training=False)
        return res

    def process_plaintext(self, plaintext: str):
        del self.model
        self.model = self.get_model(LongBert2)[0]

        tokenized = self.tokenizer.tokenize(plaintext)
        if len(tokenized) > 1536:
            tokenized = tokenized[:1536]
        sequences = self.preprocess_tokenized(tokenized)
        encoded = self.tokenizer(plaintext, max_length=1536, truncation=True, padding='max_length')

        # get the classification result
        baseline_acc = self.classify_example(
            (encoded.data['input_ids'], encoded.data['attention_mask'], encoded.data['token_type_ids']), self.model)

        # delete the model and allocate a new one so we can do the IG
        del self.model
        self.model = self.get_model(LongBert2ForIG)[0]

        # get the integrated gradients
        smooth_gradients = self.get_smooth_gradients(encoded, float(baseline_acc), num_steps=5).tolist()

        print("Baseline: " + str(float(baseline_acc)) + "\n")
        results = []
        for n in [1]:  # [1, 2, 3, 5, 7, 10, 20, 30]:
            results = results + self.process_ngrams(smooth_gradients, sequences, baseline_acc, n)

        return {"tokenized": tokenized, "results": results, "baseline": float(baseline_acc)}

    def get_noisy_examples(self, input_ids, num_steps):
        embedded = self.embedded_ids(input_ids)
        stdev = 0.15 * (np.max(embedded) - np.min(embedded))
        noisy = []

        for i in range(num_steps):
            noise = np.random.normal(0, stdev, embedded.shape).astype(np.float32)
            noisy.append(embedded + noise)

        return noisy

    def get_smooth_gradients(self, example, prediction, num_steps=2):
        # create interpolated examples
        noisy_examples = self.get_noisy_examples((example.data['input_ids']), num_steps)

        total_grads = np.zeros(shape=(1536, 768))
        # for each interpolation get embedding gradients
        for ex in noisy_examples:
            total_grads = total_grads + self.get_gradients((ex, example.data['attention_mask'], example.data['token_type_ids'])) # a gradient has shape (1536, 768)

        total_grads = total_grads / num_steps
        total_grads = total_grads * self.embedded_ids(example.data['input_ids'])
        total_grads = np.squeeze(total_grads, axis=0)
        return np.sum(total_grads, axis=1)

    def embedded_ids(self, ids):
        """
        Given input_ids vector (1536,), creates an embedded representation (1536, 768)
        :param ids:
        :return:
        """
        return tf.gather(self.embeddings, ids).numpy()

    def get_gradients(self, example):
        """
        Given an input, returns the gradients for the embedding layer
        We use the inputs_embeds argument of the TFBertModel to pass
        the embedded input directly, bypassing BERTs internal embedding
        :param example:
        :return:
        """
        example = (
        tf.expand_dims(tf.convert_to_tensor(example[0], name='inputs_embeds'), axis=0),
        tf.expand_dims(tf.convert_to_tensor(example[1], name='attention_mask'), axis=0),
        tf.expand_dims(tf.convert_to_tensor(example[2], name='token_type_ids'), axis=0)
        )
        # because the passed embeddings are not trainable variables, we need to
        # watch them and then compute the gradients w.r.t. them
        with tf.GradientTape() as tape:
            tape.watch(example[0])
            res = self.model(None, attention_mask=example[1], token_type_ids=example[2], inputs_embeds=example[0], training=False)
        grads = tape.gradient(res, example[0])
        return grads.numpy()

    def preprocess_tokenized(self, tokenized: list):
        word_indices = []
        sequences = []
        words = []
        word = ""
        for i in range(min(len(tokenized), 1534)):
            if "##" in tokenized[i]:
                word += tokenized[i][2:]
            else:
                word_indices.append(i + 1)
                words.append(word)
                word = tokenized[i]

        sequences.append(word_indices)
        return sequences

    def process_ngrams(self, array, sequences: list, baseline_acc: float, n: int):
        # token type ids all 0
        # attention mask 1 where input ids not 0
        print("Processing " + str(n) + "-grams")
        # beg_index = 1
        # end_index = encoded.data['input_ids'].index(3)
        output = []

        total_len = 0
        for sequence in sequences:
            seq_length = len(sequence)
            total_len += sequence[len(sequence) - 1]
            for i in range(seq_length):
                if i + n >= seq_length:
                    break
                if i + n == seq_length:
                    try:
                        res = 0
                        res += self.sum_range(array, sequence[i], sequence[i + n])
                    except IndexError:
                        pass
                    output.append([i, n, res])
                    break
                else:
                    res = 0
                    res += self.sum_range(array, sequence[i], sequence[i + n])
                    output.append([i, n, res])

        return output

    def sum_range(self, values, beg, end):
        res = 0
        for i in range(beg, end):
            res += values[i]
        return res

class PreprocessorSmoothGRAD:
    def __init__(self):
        model, tokenizer = self.get_model(LongBert2)
        self.tokenizer = tokenizer
        self.model = model

        # we need to get the embeddings to perform the interpolation
        self.embeddings = self.model.bert.bert.get_input_embeddings().weights[0]

    def get_model(self, model_class):
        model = model_class()
        model.load_weights("saved-weights-1")
        return model, AutoTokenizer.from_pretrained("UWB-AIR/Czert-B-base-cased")

    def classify_example(self, _input, model):
        el = (
            tf.expand_dims(tf.convert_to_tensor(_input[0]), axis=0),
            tf.expand_dims(tf.convert_to_tensor(_input[1]), axis=0),
            tf.expand_dims(tf.convert_to_tensor(_input[2]), axis=0)
        )
        res = model(el, training=False)
        return res

    def process_plaintext(self, plaintext: str):
        del self.model
        self.model = self.get_model(LongBert2)[0]

        tokenized = self.tokenizer.tokenize(plaintext)
        if len(tokenized) > 1536:
            tokenized = tokenized[:1536]
        sequences = self.preprocess_tokenized(tokenized)
        encoded = self.tokenizer(plaintext, max_length=1536, truncation=True, padding='max_length')

        # get the classification result
        baseline_acc = self.classify_example(
            (encoded.data['input_ids'], encoded.data['attention_mask'], encoded.data['token_type_ids']), self.model)

        # delete the model and allocate a new one so we can do the IG
        del self.model
        self.model = self.get_model(LongBert2ForIG)[0]

        # get the integrated gradients
        smooth_gradients = self.get_smooth_gradients(encoded, float(baseline_acc), num_steps=5).tolist()

        print("Baseline: " + str(float(baseline_acc)) + "\n")
        results = []
        for n in [1]:  # [1, 2, 3, 5, 7, 10, 20, 30]:
            results = results + self.process_ngrams(smooth_gradients, sequences, baseline_acc, n)

        return {"tokenized": tokenized, "results": results, "baseline": float(baseline_acc)}

    def get_noisy_examples(self, input_ids, num_steps):
        embedded = self.embedded_ids(input_ids)
        stdev = 0.15 * (np.max(embedded) - np.min(embedded))
        noisy = []

        for i in range(num_steps):
            noise = np.random.normal(0, stdev, embedded.shape).astype(np.float32)
            noisy.append(embedded + noise)

        return noisy

    def get_smooth_gradients(self, example, prediction, num_steps=2):
        # create interpolated examples
        noisy_examples = self.get_noisy_examples((example.data['input_ids']), num_steps)

        total_grads = np.zeros(shape=(1536, 768))
        # for each interpolation get embedding gradients
        for ex in noisy_examples:
            total_grads = total_grads + self.get_gradients((ex, example.data['attention_mask'], example.data['token_type_ids'])) # a gradient has shape (1536, 768)

        total_grads = total_grads / num_steps
        total_grads = np.squeeze(total_grads, axis=0)
        return np.sum(total_grads, axis=1)

    def embedded_ids(self, ids):
        """
        Given input_ids vector (1536,), creates an embedded representation (1536, 768)
        :param ids:
        :return:
        """
        return tf.gather(self.embeddings, ids).numpy()

    def get_gradients(self, example):
        """
        Given an input, returns the gradients for the embedding layer
        We use the inputs_embeds argument of the TFBertModel to pass
        the embedded input directly, bypassing BERTs internal embedding
        :param example:
        :return:
        """
        example = (
        tf.expand_dims(tf.convert_to_tensor(example[0], name='inputs_embeds'), axis=0),
        tf.expand_dims(tf.convert_to_tensor(example[1], name='attention_mask'), axis=0),
        tf.expand_dims(tf.convert_to_tensor(example[2], name='token_type_ids'), axis=0)
        )
        # because the passed embeddings are not trainable variables, we need to
        # watch them and then compute the gradients w.r.t. them
        with tf.GradientTape() as tape:
            tape.watch(example[0])
            res = self.model(None, attention_mask=example[1], token_type_ids=example[2], inputs_embeds=example[0], training=False)
        grads = tape.gradient(res, example[0])
        return grads.numpy()

    def preprocess_tokenized(self, tokenized: list):
        word_indices = []
        sequences = []
        words = []
        word = ""
        for i in range(min(len(tokenized), 1534)):
            if "##" in tokenized[i]:
                word += tokenized[i][2:]
            else:
                word_indices.append(i + 1)
                words.append(word)
                word = tokenized[i]

        sequences.append(word_indices)
        return sequences

    def process_ngrams(self, array, sequences: list, baseline_acc: float, n: int):
        # token type ids all 0
        # attention mask 1 where input ids not 0
        print("Processing " + str(n) + "-grams")
        # beg_index = 1
        # end_index = encoded.data['input_ids'].index(3)
        output = []

        total_len = 0
        for sequence in sequences:
            seq_length = len(sequence)
            total_len += sequence[len(sequence) - 1]
            for i in range(seq_length):
                if i + n >= seq_length:
                    break
                if i + n == seq_length:
                    try:
                        res = 0
                        res += self.sum_range(array, sequence[i], sequence[i + n])
                    except IndexError:
                        pass
                    output.append([i, n, res])
                    break
                else:
                    res = 0
                    res += self.sum_range(array, sequence[i], sequence[i + n])
                    output.append([i, n, res])

        return output

    def sum_range(self, values, beg, end):
        res = 0
        for i in range(beg, end):
            res += values[i]
        return res


class PreprocessorIntegratedGradients512:
    def __init__(self):
        model, tokenizer = self.get_model(BertClassifierForIG)
        self.tokenizer = tokenizer
        self.model = model

        # we need to get the embeddings to perform the interpolation
        self.embeddings = self.model.bert.bert.get_input_embeddings().weights[0]

    def get_model(self, model_class):
        model = model_class()
        model.load_weights("saved-weights-512")
        return model, AutoTokenizer.from_pretrained("UWB-AIR/Czert-B-base-cased")

    def split(self, _input):
        num_blocks = len(_input[1]) / 512
        embeddings = np.split(np.array(_input[0]), indices_or_sections=num_blocks, axis=0)
        att_mask = np.split(np.array(_input[1]), indices_or_sections=num_blocks, axis=0)
        tt_ids = np.split(np.array(_input[2]), indices_or_sections=num_blocks, axis=0)

        return embeddings, att_mask, tt_ids

    def classify_example(self, example):
        """
        Given an input, returns the gradients for the embedding layer
        We use the inputs_embeds argument of the TFBertModel to pass
        the embedded input directly, bypassing BERTs internal embedding
        :param example:
        :return:
        """
        split = self.split(example)
        results = []
        for i in range(len(split[0])):
            example = (
            tf.expand_dims(tf.convert_to_tensor(split[0][i], name='inputs_embeds'), axis=0),
            tf.expand_dims(tf.convert_to_tensor(split[1][i], name='attention_mask'), axis=0),
            tf.expand_dims(tf.convert_to_tensor(split[2][i], name='token_type_ids'), axis=0)
            )
            # because the passed embeddings are not trainable variables, we need to
            # watch them and then compute the gradients w.r.t. them

            results.append(float(self.model(None, attention_mask=example[1], token_type_ids=example[2], inputs_embeds=example[0], training=False)))

        return sum(results) / len(results)


    def process_plaintext(self, plaintext: str):
        tokenized = self.tokenizer.tokenize(plaintext)
        sequences = self.preprocess_tokenized(tokenized)
        encoded = self.tokenizer(plaintext, truncation=True, padding=True, pad_to_multiple_of=512)


        # get the classification result
        baseline_acc = self.classify_example(
            (self.embedded_ids(encoded.data['input_ids']), encoded.data['attention_mask'], encoded.data['token_type_ids']))

        # get the integrated gradients
        integrated_gradients = self.get_integrated_gradients(encoded, float(baseline_acc), num_steps=100).numpy().tolist()
        print("attribution sum: " + str(sum(integrated_gradients)))

        #self.create_vis()

        abssum = 0
        for el in integrated_gradients:
            abssum += abs(el)
        print("attribution sum abs: " + str(abssum) )

        possum = 0
        negsum = 0
        for el in integrated_gradients:
            if el > 0:
                possum += el
            else:
                negsum += el

        print("attribution sum pos: " + str(possum))
        print("attribution sum neg: " + str(negsum))

        print("Baseline: " + str(float(baseline_acc)) + "\n")
        results = []
        for n in [1]:  # [1, 2, 3, 5, 7, 10, 20, 30]:
            results = results + self.process_ngrams(integrated_gradients, sequences, baseline_acc, n)

        return {"tokenized": tokenized, "results": results, "baseline": float(baseline_acc)}

    def create_vis(self):
        with open("predictions.csv", "w+") as f:
            for i in range(len(self.saved_predictions)):
                f.write(str(i) + ";" + str(float(self.saved_predictions[i])) + "\n")

        for i in range(len(self.saved_gradients)):
            self.saved_gradients[i] = tf.squeeze(self.saved_gradients[i], axis=0)
            self.saved_gradients[i] = tf.abs(self.saved_gradients[i])
            self.saved_gradients[i] = tf.reduce_mean(self.saved_gradients[i])

        with open("gradients.csv", "w+") as f:
            for i in range(len(self.saved_gradients)):
                f.write(str(i) + ";" + str(float(self.saved_gradients[i])) + "\n")

    def get_baseline(self, ids, att_mask, tt_ids):
        length = ids.index(3)
        res = 0
        while abs(res - 0.5) > 0.075:
            baseline = [2] + np.random.randint(low=5, high=30000, size=(length - 2,)).tolist() + [3] + [0 for _ in range(len(ids) - length)]
            res = self.classify_example((self.embedded_ids(baseline), att_mask, tt_ids))

        baseline = self.embedded_ids(baseline)
        return baseline

    def get_integrated_gradients(self, example, prediction, num_steps=75):
        # create baseline
        baseline = self.get_baseline(example.data['input_ids'], example.data['attention_mask'], example.data['token_type_ids'])

        # create interpolated examples
        interpolated_examples = self.get_interpolated_examples((example.data['input_ids'], example.data['attention_mask'], example.data['token_type_ids']), num_steps, baseline)

        # we need the embedded ids for computations later on
        input_ids = np.array(self.embedded_ids(example.data['input_ids']))
        # we sum them over the embedding axis to get a (1536,) tensor

        grads = []
        # for each interpolation get embedding gradients
        for interpolated in interpolated_examples:
            grads.append(self.get_gradients(interpolated)) # a gradient has shape (1536, 768)
        for i in range(len(grads)):
            grad = tf.convert_to_tensor(grads[i])  # to tensor
            #grad = tf.squeeze(grad, axis=0)
            #grad = tf.reduce_sum(grad, axis=1)    # average over axis 1 to shape (1536,)
            grads[i] = grad

        grads = tf.convert_to_tensor(grads, dtype=tf.float32) # create tensor of size (num_steps + 1, 1536)

        grads = (grads[:-1] + grads[1:]) / 2.0      # approximate with trapezoidal rule, shape = (num_steps, 1536)
        avg_grads = tf.reduce_mean(grads, axis=0)   # average over samples, shape = (1536)

        integrated_grads = (input_ids - np.array(baseline)) * avg_grads   # calculate integral, output shape = (1536)
        integrated_grads = tf.reduce_sum(integrated_grads, axis=1)
        return integrated_grads

    def get_interpolated_examples(self, example, num_steps, baseline=None):
        """
        Given a baseline and target, creates a n-step linear interpolation between them
        :param example: array of shape (1536, 768)
        :param num_steps: number of interpolation steps
        :param baseline: baseline of shape (1536, 768)
        :return:
        """
        ids = example[0]
        interpolated_examples = []

        # embed the input_ids
        ids = np.array(self.embedded_ids(ids))
        # do the interpolation
        for step in range(num_steps + 1):
            interpolated = baseline + (step / num_steps) * (ids - baseline)
            interpolated_examples.append((interpolated.tolist(), example[1], example[2]))

        return interpolated_examples

    def embedded_ids(self, ids):
        """
        Given input_ids vector (1536,), creates an embedded representation (1536, 768)
        :param ids:
        :return:
        """
        return tf.gather(self.embeddings, ids).numpy().tolist()

    saved_gradients = []
    saved_predictions = []

    def get_gradients(self, example):
        """
        Given an input, returns the gradients for the embedding layer
        We use the inputs_embeds argument of the TFBertModel to pass
        the embedded input directly, bypassing BERTs internal embedding
        :param example:
        :return:
        """
        split = self.split(example)
        results = []
        for i in range(len(split[0])):
            example = (
            tf.expand_dims(tf.convert_to_tensor(split[0][i], name='inputs_embeds'), axis=0),
            tf.expand_dims(tf.convert_to_tensor(split[1][i], name='attention_mask'), axis=0),
            tf.expand_dims(tf.convert_to_tensor(split[2][i], name='token_type_ids'), axis=0)
            )
            # because the passed embeddings are not trainable variables, we need to
            # watch them and then compute the gradients w.r.t. them
            with tf.GradientTape() as tape:
                tape.watch(example[0])
                res = self.model(None, attention_mask=example[1], token_type_ids=example[2], inputs_embeds=example[0], training=False)
            grads = tape.gradient(res, example[0])
            results.append(tf.squeeze(grads, axis=0).numpy().tolist())

        output = np.concatenate(results, axis=0)
        return output

    def preprocess_tokenized(self, tokenized: list):
        word_indices = []
        sequences = []
        words = []
        word = ""
        for i in range(min(len(tokenized), 1534)):
            if "##" in tokenized[i]:
                word += tokenized[i][2:]
            else:
                word_indices.append(i + 1)
                words.append(word)
                word = tokenized[i]

        sequences.append(word_indices)
        return sequences

    def process_ngrams(self, array, sequences: list, baseline_acc: float, n: int):
        # token type ids all 0
        # attention mask 1 where input ids not 0
        print("Processing " + str(n) + "-grams")
        # beg_index = 1
        # end_index = encoded.data['input_ids'].index(3)
        output = []

        total_len = 0
        for sequence in sequences:
            seq_length = len(sequence)
            total_len += sequence[len(sequence) - 1]
            for i in range(seq_length):
                if i + n >= seq_length:
                    break
                if i + n == seq_length:
                    try:
                        res = 0
                        res += self.sum_range(array, sequence[i], sequence[i + n])
                    except IndexError:
                        pass
                    output.append([i, n, res])
                    break
                else:
                    res = 0
                    res += self.sum_range(array, sequence[i], sequence[i + n])
                    output.append([i, n, res])

        return output

    def sum_range(self, values, beg, end):
        res = 0
        for i in range(beg, end):
            res += values[i]
        return res


# ----------------------------------------------------------------------------- PREPROCESSOR INTEGRATED GRADIENTS
# Custom IG attribution implementation
# This works, unlike the library, but we need to use the basic LongBert2 model
# to get the classification result and then the LongBert2ForIG to be able to
# compute the gradients

class PreprocessorIntegratedGradients:
    def __init__(self):
        model, tokenizer = self.get_model(LongBert2)
        self.tokenizer = tokenizer
        self.model = model

        # we need to get the embeddings to perform the interpolation
        self.embeddings = self.model.bert.bert.get_input_embeddings().weights[0]

    def get_model(self, model_class):
        model = model_class()
        model.load_weights("saved-weights-1")
        return model, AutoTokenizer.from_pretrained("UWB-AIR/Czert-B-base-cased")

    def classify_example(self, _input, model):
        el = (
            tf.expand_dims(tf.convert_to_tensor(_input[0]), axis=0),
            tf.expand_dims(tf.convert_to_tensor(_input[1]), axis=0),
            tf.expand_dims(tf.convert_to_tensor(_input[2]), axis=0)
        )
        res = model(el, training=False)
        return res

    def process_plaintext(self, plaintext: str):
        del self.model
        self.model = self.get_model(LongBert2)[0]

        tokenized = self.tokenizer.tokenize(plaintext)
        if len(tokenized) > 1536:
            tokenized = tokenized[:1536]
        sequences = self.preprocess_tokenized(tokenized)
        encoded = self.tokenizer(plaintext, max_length=1536, truncation=True, padding='max_length')

        # get the classification result
        baseline_acc = self.classify_example(
            (encoded.data['input_ids'], encoded.data['attention_mask'], encoded.data['token_type_ids']), self.model)

        # delete the model and allocate a new one so we can do the IG
        del self.model
        self.model = self.get_model(LongBert2ForIG)[0]

        # get the integrated gradients
        integrated_gradients = self.get_integrated_gradients(encoded, float(baseline_acc), num_steps=2).numpy().tolist()
        print("attribution sum: " + str(sum(integrated_gradients)))

        abssum = 0
        for el in integrated_gradients:
            abssum += abs(el)
        print("attribution sum abs: " + str(abssum) )

        possum = 0
        negsum = 0
        for el in integrated_gradients:
            if el > 0:
                possum += el
            else:
                negsum += el

        print("attribution sum pos: " + str(possum))
        print("attribution sum neg: " + str(negsum))

        print("Baseline: " + str(float(baseline_acc)) + "\n")
        results = []
        for n in [1]:  # [1, 2, 3, 5, 7, 10, 20, 30]:
            results = results + self.process_ngrams(integrated_gradients, sequences, baseline_acc, n)

        return {"tokenized": tokenized, "results": results, "baseline": float(baseline_acc)}

    def get_baseline(self, ids, att_mask, tt_ids):
        # generate a PAD baseline for the interpolation
        # baseline = np.array([[0 for __ in range(768)] for _ in ids])
        length = ids.index(3) + 1
        nearest_50 = 50 * round(length / 50)
        if nearest_50 < 100:
            nearest_50 = 100
        elif nearest_50 > 1500:
            nearest_50 = 1500

        baseline = json.load(open("baselines/" + str(nearest_50), "r", encoding='utf-8'))

        del self.model
        self.model = self.get_model(LongBert2)[0]

        res = self.classify_example((baseline, att_mask, tt_ids), self.model)
        if abs(res - 0.5) > 0.075:
            while abs(res - 0.5) > 0.075:
                baseline = [2] + np.random.randint(low=5, high=30000, size=(nearest_50 - 2,)).tolist() + [3] + [0 for _ in range(1536 - nearest_50)]
                res = self.classify_example((baseline, att_mask, tt_ids), self.model)

        del self.model
        self.model = self.get_model(LongBert2ForIG)[0]
        baseline = self.embedded_ids(baseline)
        return baseline

    def get_integrated_gradients(self, example, prediction, num_steps=2):
        # create baseline
        baseline = self.get_baseline(example.data['input_ids'], example.data['attention_mask'], example.data['token_type_ids'])

        # create interpolated examples
        interpolated_examples = self.get_interpolated_examples((example.data['input_ids'], example.data['attention_mask'], example.data['token_type_ids']), num_steps, baseline)

        # we need the embedded ids for computations later on
        input_ids = np.array(self.embedded_ids(example.data['input_ids']))
        # we sum them over the embedding axis to get a (1536,) tensor

        grads = []
        # for each interpolation get embedding gradients
        for interpolated in interpolated_examples:
            grads.append(self.get_gradients(interpolated)) # a gradient has shape (1536, 768)
        for i in range(len(grads)):
            grad = tf.convert_to_tensor(grads[i])  # to tensor
            grad = tf.squeeze(grad, axis=0)
            #grad = tf.reduce_sum(grad, axis=1)    # average over axis 1 to shape (1536,)
            grads[i] = grad

        grads = tf.convert_to_tensor(grads, dtype=tf.float32) # create tensor of size (num_steps + 1, 1536)

        grads = (grads[:-1] + grads[1:]) / 2.0      # approximate with trapezoidal rule, shape = (num_steps, 1536)
        avg_grads = tf.reduce_mean(grads, axis=0)   # average over samples, shape = (1536)

        integrated_grads = (input_ids - np.array(baseline)) * avg_grads   # calculate integral, output shape = (1536)
        integrated_grads = tf.reduce_sum(integrated_grads, axis=1)
        return integrated_grads

    def get_interpolated_examples(self, example, num_steps, baseline=None):
        """
        Given a baseline and target, creates a n-step linear interpolation between them
        :param example: array of shape (1536, 768)
        :param num_steps: number of interpolation steps
        :param baseline: baseline of shape (1536, 768)
        :return:
        """
        ids = example[0]
        interpolated_examples = []

        # embed the input_ids
        ids = np.array(self.embedded_ids(ids))
        # do the interpolation
        for step in range(num_steps + 1):
            interpolated = baseline + (step / num_steps) * (ids - baseline)
            interpolated_examples.append((interpolated.tolist(), example[1], example[2]))

        return interpolated_examples

    def embedded_ids(self, ids):
        """
        Given input_ids vector (1536,), creates an embedded representation (1536, 768)
        :param ids:
        :return:
        """
        return tf.gather(self.embeddings, ids).numpy().tolist()

    def get_gradients(self, example):
        """
        Given an input, returns the gradients for the embedding layer
        We use the inputs_embeds argument of the TFBertModel to pass
        the embedded input directly, bypassing BERTs internal embedding
        :param example:
        :return:
        """
        example = (
        tf.expand_dims(tf.convert_to_tensor(example[0], name='inputs_embeds'), axis=0),
        tf.expand_dims(tf.convert_to_tensor(example[1], name='attention_mask'), axis=0),
        tf.expand_dims(tf.convert_to_tensor(example[2], name='token_type_ids'), axis=0)
        )
        # because the passed embeddings are not trainable variables, we need to
        # watch them and then compute the gradients w.r.t. them
        with tf.GradientTape() as tape:
            tape.watch(example[0])
            res = self.model(None, attention_mask=example[1], token_type_ids=example[2], inputs_embeds=example[0], training=False)
        grads = tape.gradient(res, example[0])
        return grads.numpy().tolist()

    def preprocess_tokenized(self, tokenized: list):
        word_indices = []
        sequences = []
        words = []
        word = ""
        for i in range(min(len(tokenized), 1534)):
            if "##" in tokenized[i]:
                word += tokenized[i][2:]
            else:
                word_indices.append(i + 1)
                words.append(word)
                word = tokenized[i]

        sequences.append(word_indices)
        return sequences

    def process_ngrams(self, array, sequences: list, baseline_acc: float, n: int):
        # token type ids all 0
        # attention mask 1 where input ids not 0
        print("Processing " + str(n) + "-grams")
        # beg_index = 1
        # end_index = encoded.data['input_ids'].index(3)
        output = []

        total_len = 0
        for sequence in sequences:
            seq_length = len(sequence)
            total_len += sequence[len(sequence) - 1]
            for i in range(seq_length):
                if i + n >= seq_length:
                    break
                if i + n == seq_length:
                    try:
                        res = 0
                        res += self.sum_range(array, sequence[i], sequence[i + n])
                    except IndexError:
                        pass
                    output.append([i, n, res])
                    break
                else:
                    res = 0
                    res += self.sum_range(array, sequence[i], sequence[i + n])
                    output.append([i, n, res])

        return output

    def sum_range(self, values, beg, end):
        res = 0
        for i in range(beg, end):
            res += values[i]
        return res


# ----------------------------------------------------------------------------------------- PREPROCESSOR ATTENTION
# We calculate the attributions for each word based on the attention given to the word
# and the gradient of it's embedding vector

class PreprocessorAttention:
    def __init__(self):
        model, tokenizer = self.get_model(LongBert2ForAttention)
        self.model = model
        self.tokenizer = tokenizer

    def get_model(self, model_class):
        model = model_class()
        model.load_weights("saved-weights-1")
        return model, AutoTokenizer.from_pretrained("UWB-AIR/Czert-B-base-cased")

    def classify_example(self, _input, model, cls_token_index=None):
        el = (
            tf.expand_dims(tf.convert_to_tensor(_input[0]), axis=0),
            tf.expand_dims(tf.convert_to_tensor(_input[1]), axis=0),
            tf.expand_dims(tf.convert_to_tensor(_input[2]), axis=0)
        )
        with tf.GradientTape() as tape:
            if cls_token_index is not None:
                res = model(el, training=False, cls_token_index=cls_token_index)
            else:
                res = model(el, training=False)
        if isinstance(model, LongBert2ForAttention):
            return res
        else:
            return float(res), tape.gradient(res, model.trainable_weights)

    def process_plaintext(self, plaintext: str):
        del self.model
        self.model = self.get_model(LongBert2ForAttention)[0]

        tokenized = self.tokenizer.tokenize(plaintext)
        sequences = self.preprocess_tokenized(tokenized)
        encoded = self.tokenizer(plaintext, max_length=1536, truncation=True, padding='max_length')

        # we need to be aware of the position of the CLS token, since it might be in the middle of a 512 block
        cls_token_index = encoded.data['input_ids'].index(3)

        # get the attentions (we initialize with the LongBert2ForAttention)
        attention = self.classify_example(
            (encoded.data['input_ids'], encoded.data['attention_mask'], encoded.data['token_type_ids']), self.model, cls_token_index)

        # remove the model and allocate a new one for the gradients
        del self.model
        self.model = self.get_model(LongBert2)[0]
        # get the classification result and gradients
        baseline_acc, grads = self.classify_example(
            (encoded.data['input_ids'], encoded.data['attention_mask'], encoded.data['token_type_ids']), self.model)

        # get the embedding vectors gradients (1536, 768)
        embeddings = grads[0].values
        # for each embedded token, sum the gradients (1536, 768) -> (1536,)
        embeddings = tf.reduce_sum(embeddings, axis=1)
        # multiply the gradients and the attentions (attentions are always positive or 0)
        res = tf.multiply(embeddings, tf.convert_to_tensor(attention)).numpy().tolist()

        print("Baseline: " + str(baseline_acc) + "\n")
        results = []
        for n in [1]:
            results = results + self.process_ngrams(res, sequences, n)

        return {"tokenized": tokenized, "results": results, "baseline": baseline_acc}

    def preprocess_tokenized(self, tokenized: list):
        """
        Creates words from the tokens
        Returns a list of indices, at which words begin
        :param tokenized:
        :return:
        """
        word_indices = []
        sequences = []
        words = []
        word = ""
        for i in range(min(len(tokenized), 1534)):
            if "##" in tokenized[i]:
                word += tokenized[i][2:]
            else:
                word_indices.append(i + 1)
                words.append(word)
                word = tokenized[i]

        sequences.append(word_indices)
        return sequences

    def process_ngrams(self, array, sequences: list, n: int):
        print("Processing " + str(n) + "-grams")
        output = []

        # for n-grams, we simply sum the attributions for the tokens, that make the n-gram

        total_len = 0
        for sequence in sequences:
            seq_length = len(sequence)
            total_len += sequence[len(sequence) - 1]
            for i in range(seq_length):
                if i + n >= seq_length:
                    break
                if i + n == seq_length:
                    try:
                        res = 0
                        res += self.sum_range(array, sequence[i], sequence[i + n])
                    except IndexError:
                        pass
                    output.append([i, n, res])
                    break
                else:
                    res = 0
                    res += self.sum_range(array, sequence[i], sequence[i + n])
                    output.append([i, n, res])

        return output

    def sum_range(self, values, beg, end):
        res = 0
        for i in range(beg, end):
            res += values[i]
        return res


# ----------------------------------------------------------------------------------------- PREPROCESSOR GRADIENT
# We calculate the attributions based on the gradients of the embedding vectors,
# token_type_ids and attention mask
# since the each of the token has a (768,) vector, we sum the gradients for the vector
# We use the basic LongBert2 model, since we only need to wrap the call in GradientTape

class PreprocessorGradient:
    def __init__(self):
        model, tokenizer = self.get_model()
        self.model = model
        self.tokenizer = tokenizer

    def get_model(self):
        model = LongBert2()
        model.load_weights("saved-weights-1")
        return model, AutoTokenizer.from_pretrained("UWB-AIR/Czert-B-base-cased")

    def classify_example(self, _input, model):
        """
        Classifies the passed example and returns the result and the computed
        gradients
        :param _input:
        :param model:
        :return:
        """
        el = (
            tf.expand_dims(tf.convert_to_tensor(_input[0]), axis=0),
            tf.expand_dims(tf.convert_to_tensor(_input[1]), axis=0),
            tf.expand_dims(tf.convert_to_tensor(_input[2]), axis=0)
        )
        with tf.GradientTape() as tape:
            res = model(el, training=False)
        return float(res), tape.gradient(res, model.trainable_weights)

    def process_plaintext(self, plaintext: str):
        """
        Given plaintext, computes the attributions of n-grams to the overall result
        :param plaintext:
        :return:
        """
        tokenized = self.tokenizer.tokenize(plaintext)
        sequences = self.preprocess_tokenized(tokenized)
        encoded = self.tokenizer(plaintext, max_length=1536, truncation=True, padding='max_length')

        # get the classification result and gradients
        baseline_acc, grads = self.classify_example(
            (encoded.data['input_ids'], encoded.data['attention_mask'], encoded.data['token_type_ids']), self.model)

        # extract the gradients for embedding vectors, token_type_ids and attention_mask
        embeddings = grads[0].values
        em = self.model.bert.bert.get_input_embeddings().weights[0]
        emids = tf.gather(em, encoded.data['input_ids'])
        embeddings = tf.multiply(embeddings, emids)
        embeddings = tf.reduce_sum(embeddings, axis=1).numpy().tolist()

        print("Baseline: " + str(baseline_acc) + "\n")
        results = []
        # calculate n-gram attributions
        for n in [1]:  # [1, 2, 3, 5, 7, 10, 20, 30]:
            results = results + self.process_ngrams(embeddings, sequences, n)

        return {"tokenized": tokenized, "results": results, "baseline": baseline_acc}

    def preprocess_tokenized(self, tokenized: list):
        """
        Creates words from the tokens
        Returns a list of indices, at which words begin
        :param tokenized:
        :return:
        """
        word_indices = []
        sequences = []
        words = []
        word = ""
        for i in range(min(len(tokenized), 1534)):
            if "##" in tokenized[i]:
                word += tokenized[i][2:]
            else:
                word_indices.append(i + 1)
                words.append(word)
                word = tokenized[i]

        sequences.append(word_indices)
        return sequences

    def process_ngrams(self, embeddings, sequences: list, n: int):
        print("Processing " + str(n) + "-grams")
        output = []
        total_len = 0
        for sequence in sequences:
            seq_length = len(sequence)
            total_len += sequence[len(sequence) - 1]

            # for each n-gram, we sum the gradients
            for i in range(seq_length):
                if i + n >= seq_length:
                    break
                if i + n == seq_length:
                    try:
                        res = 0
                        res += self.sum_range(embeddings, sequence[i], sequence[i + n])
                    except IndexError:
                        pass
                    output.append([i, n, res])
                    break
                else:
                    res = 0
                    res += self.sum_range(embeddings, sequence[i], sequence[i + n])
                    output.append([i, n, res])

        return output

    def sum_range(self, values, beg, end):
        """
        given list of gradient vectors, sum the vectors at indices [beg, end)
        :param values:
        :param beg:
        :param end:
        :return:
        """
        res = 0
        for i in range(beg, end):
            res += values[i]
        return res


# ----------------------------------------------------------------------------------------- PREPROCESSOR COVERING
# Basic attribution method, which covers n-grams of words to see how the resulting
# classification changes
# in general not useful

class PreprocessorCovering:
    def __init__(self):
        model, tokenizer = self.get_model()
        self.model = model
        self.tokenizer = tokenizer

    def get_model(self):
        """
        Loads model and tokenizer
        :return:
        """
        model = LongBert2()
        model.load_weights("saved-weights-1")
        return model, AutoTokenizer.from_pretrained("UWB-AIR/Czert-B-base-cased")

    def classify_example(self, _input, model):
        """
        Classifies an example and returns a result
        :param _input:
        :param model:
        :return:
        """
        el = (
            tf.expand_dims(tf.convert_to_tensor(_input[0]), axis=0),
            tf.expand_dims(tf.convert_to_tensor(_input[1]), axis=0),
            tf.expand_dims(tf.convert_to_tensor(_input[2]), axis=0)
        )
        res = model(el, training=False)
        return float(res)

    def process_plaintext(self, plaintext: str):
        """
        Given plaintext, computes the attributions of n-grams to the overall result
        :param plaintext:
        :return:
        """
        tokenized = self.tokenizer.tokenize(plaintext)
        sequences = self.preprocess_tokenized(tokenized)
        encoded = self.tokenizer(plaintext, max_length=1536, truncation=True, padding='max_length')
        baseline_acc = self.classify_example(
            (encoded.data['input_ids'], encoded.data['attention_mask'], encoded.data['token_type_ids']), self.model)

        print("Baseline: " + str(baseline_acc) + "\n")
        results = []
        # go trough n-grams
        for n in [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]:
            results = results + self.process_ngrams(encoded, sequences, self.model, baseline_acc, n)

        return {"tokenized": tokenized, "results": results, "baseline": baseline_acc}

    def preprocess_tokenized(self, tokenized: list):
        """
        Creates words from the tokens
        Returns a list of indices, at which words begin
        :param tokenized:
        :return:
        """
        word_indices = []
        sequences = []
        words = []
        word = ""
        for i in range(min(len(tokenized), 1534)):
            if "##" in tokenized[i]:
                word += tokenized[i][2:]
            else:
                word_indices.append(i + 1)
                words.append(word)
                word = tokenized[i]

        sequences.append(word_indices)
        return sequences

    def process_ngrams(self, encoded, sequences: list, model: TFBertModel, baseline_acc: float, n: int):
        """
        Calculates attributions for all the n-grams
        :param encoded:
        :param sequences:
        :param model:
        :param baseline_acc:
        :param n:
        :return:
        """
        # token type ids all 0
        # attention mask 1 where input ids not 0
        print("Processing " + str(n) + "-grams")
        # beg_index = 1
        # end_index = encoded.data['input_ids'].index(3)
        output = []

        total_len = 0
        for sequence in sequences:
            seq_length = len(sequence)
            total_len += sequence[len(sequence) - 1]

            # shady index stuff but it works
            # basically we splice out the input_ids of the word and add PAD tokens to the end
            # we have to compensate for it in the attention_mask, so we remove a few 1s and add 0s
            for i in range(seq_length):
                if i + n > seq_length:
                    break
                if i + n == seq_length:
                    modified = [encoded.data['input_ids'].copy(), encoded.data['attention_mask'].copy(),
                                encoded.data['token_type_ids'].copy()]
                    modified[0] = modified[0][0:sequence[i]] + modified[0][sequence[i + n - 1] + 1:] + \
                                    [0 for _ in range(sequence[seq_length - 1] - sequence[i] + 1)]

                    modified[1] = modified[1][0:sequence[i]] + modified[0][sequence[i + n - 1] + 1:] + \
                                    [0 for _ in range(sequence[seq_length - 1] - sequence[i] + 1)]

                    res = self.classify_example(modified, model)
                    output.append([i, n, res - baseline_acc])
                    break
                else:
                    modified = [encoded.data['input_ids'].copy(), encoded.data['attention_mask'].copy(),
                                encoded.data['token_type_ids'].copy()]
                    modified[0] = modified[0][0:sequence[i]] + modified[0][sequence[i + n]:] + [0 for _ in range(
                        sequence[i + n] - sequence[i])]
                    modified[1] = modified[1][0:sequence[i]] + modified[1][sequence[i + n]:] + [0 for _ in range(
                        sequence[i + n] - sequence[i])]
                    res = self.classify_example(modified, model)
                    output.append([i, n, res - baseline_acc])

        return output


# ------------------------------------------------------------------------------------------- APPLICATION CONTROL

def control_batch(text):
    """
    If the input file is specified at start
    :param text:
    :return:
    """
    _dir = "articles_to_vis"
    output_dir = "articles_vis/gradients_x_values"
    for file in os.listdir(_dir):
        with open(_dir + "/" + file, "r", encoding='utf-8') as f:
            text = f.read()
            preprocessor = PreprocessorGradient()
            data = preprocessor.process_plaintext(text)
            with open(output_dir + "/" + file + ".json", "w+", encoding='utf-8') as of:
                of.write(json.dumps(data))


def control_interactive():
    """
    GUI
    :return:
    """
    layout = [
        [
            [PySimpleGUI.Text("Zadejte text pro klasifikaci")],
            [PySimpleGUI.Multiline(size=(60, 20), key='textbox')],
            [PySimpleGUI.Button("Zpracovat"), PySimpleGUI.Text("Zpracování vstupu chvíli zabere", key="statustext"),
             PySimpleGUI.Button("Smazat")]
        ]
    ]
    window = PySimpleGUI.Window(title="Demo", layout=layout)
    preprocessor = PreprocessorGradient()
    while True:
        event, values = window.read()
        if event is None:
            exit(0)
        elif "Zpracovat" in event:
            text = values['textbox']
            data = preprocessor.process_plaintext(insert_paragraphs(text))
            WebViewWindowWords(data)
        elif "Smazat" in event:
            window['textbox'].update("")


def insert_paragraphs(text):
    """
    Detect newlines and insert <p></p>
    :param text:
    :return:
    """
    split = text.split("\n")
    text = "<p>"
    for i in range(len(split) - 1):
        if split[i] == "":
            continue
        text += split[i] + "</p><p>"
    text += split[len(split) - 1] + "</p>"
    return text


def run(file):
    """
    Main
    :param file:
    :return:
    """

    if file is not None:
        control_batch(None)
    else:
        control_interactive()


if __name__ == "__main__":
    run(None)
