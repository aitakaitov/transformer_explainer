import tensorflow as tf
import numpy as np
from transformer_explainer.utils import utils
from transformer_explainer.utils.explanation import Explanation


class ExplainerGradients:
    def __init__(self, model):
        self.model = model
        self.input_classification = 0
        self.x_inputs = False
        self.input_embeds = None
        self.silent = True

    def _classify(self, example):
        return float(self.model(utils.to_tensor_and_expand(example)))

    def _get_gradients(self, example):
        if not self.silent:
            print("[ExplainerGradients] Calculating gradients")
        example = utils.to_tensor_and_expand(example)
        with tf.GradientTape() as tape:
            tape.watch(example[0])
            res = self.model(example)

        grads = tape.gradient(res, example[0])
        grads = tf.squeeze(grads, axis=0)

        if self.x_inputs:
            grads = grads * self.input_embeds

        return grads

    def explain(self, input_embeds: np.ndarray, attention_mask: np.ndarray, token_type_ids: np.ndarray,
                tokens: list, x_inputs=False, silent=True):
        self.silent = silent
        if not self.silent:
            print("[ExplainerGradients] Classifying input")
        self.input_embeds = input_embeds
        self.x_inputs = x_inputs

        if x_inputs:
            if not self.silent:
                print("[ExplainerGradients] x_inputs passed")

        self.input_classification = self._classify((input_embeds, attention_mask, token_type_ids))
        gradients = self._get_gradients((input_embeds, attention_mask, token_type_ids))
        gradients = tf.reduce_sum(gradients, axis=1)
        return Explanation(tokens, gradients.numpy().tolist(), self.input_classification)
