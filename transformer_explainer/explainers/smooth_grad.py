import tensorflow as tf
import numpy as np
from transformer_explainer.utils import utils
from transformer_explainer.utils.explanation import Explanation


class ExplainerSmoothGRAD:
    def __init__(self, model):
        self.model = model
        self.input_classification = 0
        self.input_embeds = None
        self.x_inputs = False
        self.silent = True

    def _classify(self, example):
        return float(self.model(utils.to_tensor_and_expand(example)))

    def _get_gradients(self, example):
        example = utils.to_tensor_and_expand(example)
        with tf.GradientTape() as tape:
            tape.watch(example[0])
            res = self.model(example)

        grads = tape.gradient(res, example[0])
        return grads

    def explain(self, input_embeds: np.ndarray, attention_mask: np.ndarray, token_type_ids: np.ndarray,
                tokens: list, noise_size=0.0125, steps=50, x_inputs=False, silent=True):
        self.silent = silent
        if not silent:
            print("[ExplainerSmoothGRAD] Classifying input")
        self.input_embeds = input_embeds
        self.x_inputs = x_inputs

        if x_inputs:
            if not silent:
                print("[ExplainerSmoothGRAD] x_inputs passed")

        self.input_classification = self._classify((input_embeds, attention_mask, token_type_ids))
        noisy_examples = self._generate_examples(input_embeds, attention_mask, token_type_ids,
                                                        steps, noise_size)
        smooth_gradients = self._get_smooth_gradients(noisy_examples)
        return Explanation(tokens, smooth_gradients.numpy().tolist(), self.input_classification)

    def _get_smooth_gradients(self, examples):
        if not self.silent:
            print("[ExplainerSmoothGRAD] Calculating gradients")
        gradient_sum = tf.zeros(shape=examples[0][0].shape)
        for ex in examples:
            gradient_sum += tf.squeeze(self._get_gradients(ex), axis=0)

        gradient_sum /= len(examples)

        if self.x_inputs:
            gradient_sum = gradient_sum * self.input_embeds

        gradient_sum = tf.reduce_sum(gradient_sum, axis=1)
        return gradient_sum

    def _generate_examples(self, input_embeds, attention_mask, token_type_ids, steps, noise_size):
        if not self.silent:
            print("[ExplainerSmoothGRAD] Generating " + str(steps) + " noisy samples")
        stdev = noise_size * (np.max(input_embeds) - np.min(input_embeds))
        noisy = []
        for i in range(steps):
            noise = np.random.normal(0, stdev, input_embeds.shape).astype(np.float32)
            noisy.append((input_embeds + noise, attention_mask, token_type_ids))

        return noisy
