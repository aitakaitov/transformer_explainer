import tensorflow as tf
from transformers import AutoTokenizer, TFAutoModel, TFBertModel
import numpy as np


class Bert1536(tf.keras.Model):
    def __init__(self):
        super(Bert1536, self).__init__()
        self.bert = TFAutoModel.from_pretrained("UWB-AIR/Czert-B-base-cased")
        self.bert.config.attention_probs_dropout_prob = 0.1
        self.bert.config.hidden_dropout_prob = 0.1
        self.lstm = tf.keras.layers.RNN(tf.keras.layers.LSTMCell(100, dropout=0.0, recurrent_dropout=0.0))
        self.dropout_1 = tf.keras.layers.Dropout(0.0)
        self.dense_1 = tf.keras.layers.Dense(24, activation='relu', name='classifier')
        self.dropout_2 = tf.keras.layers.Dropout(0.0)
        self.dense_output = tf.keras.layers.Dense(1, activation='sigmoid', name='output', dtype='float32')

    @staticmethod
    def get_tokenizer():
        return AutoTokenizer.from_pretrained("UWB-AIR/Czert-B-base-cased")

    def call(self, inputs, training=False, output_attentions=False):
        # split data into 512 blocks
        x = self.split_data(inputs)
        # x = eight blocks of shape (batch, 512) for each of the three inputs

        # process each of the blocks
        # for each input (batch, 512) we get a (batch, 768) result
        results = []
        for block in x:
            if not output_attentions:
                res = self.bert(None, inputs_embeds=block[0], attention_mask=block[1], token_type_ids=block[2], training=training).pooler_output
                results.append(res)
            else:
                res = self.bert(None, inputs_embeds=block[0], attention_mask=block[1], token_type_ids=block[2],
                                training=training, output_attentions=True).attentions
                res = tf.reduce_mean(res, axis=0)
                res = tf.squeeze(res)
                res = tf.reduce_mean(res, axis=0)
                res = tf.gather(res, 0)
                results.append(res)

        if not output_attentions:
            concatenated = tf.stack(results, axis=1)
        else:
            return tf.concat(results, axis=-1)

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
