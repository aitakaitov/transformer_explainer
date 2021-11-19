import tensorflow as tf
import numpy as np
from transformer_explainer.utils import utils
from transformer_explainer.utils.explanation import Explanation


class ExplainerAttention:
    def __init__(self, model):
        self.model = model
        self.input_classification = 0
        self.silent = True

    def _classify(self, example):
        return float(self.model(utils.to_tensor_and_expand(example)))

    def _get_gradients(self, example, do_sum=True):
        print("[ExplainerGradients] Calculating gradients")
        example = utils.to_tensor_and_expand(example)
        with tf.GradientTape() as tape:
            tape.watch(example[0])
            res = self.model(example)

        grads = tape.gradient(res, example[0])
        grads = tf.squeeze(grads, axis=0)

        if do_sum:
            grads = tf.reduce_sum(grads, axis=1)

        return grads

    def _get_attentions(self, example):
        example = utils.to_tensor_and_expand(example)
        attentions = self.model(example, output_attentions=True)
        return attentions

    def explain(self, input_embeds: np.ndarray, attention_mask: np.ndarray, token_type_ids: np.ndarray,
                tokens: list, method='grad', steps=40, noise_size=0.0125, silent=True):
        self.silent = silent
        if not silent:
            print("[ExplainerGradients] Classifying input")
        self.input_classification = self._classify((input_embeds, attention_mask, token_type_ids))

        if method == 'grad':
            gradients = self._get_gradients((input_embeds, attention_mask, token_type_ids), do_sum=True)
            gradients = tf.math.sign(gradients)
        elif method == 'smoothgrad':
            examples = self._generate_examples(input_embeds=input_embeds, attention_mask=attention_mask,
                                               token_type_ids=token_type_ids, steps=steps, noise_size=noise_size)
            gradients = self._get_smooth_gradients(examples)
            gradients = tf.math.sign(gradients)
        else:
            pass

        attentions = self._get_attentions((input_embeds, attention_mask, token_type_ids))
        attributions = attentions * gradients
        return Explanation(tokens, attributions.numpy().tolist(), self.input_classification)

    def _get_smooth_gradients(self, examples):
        if not self.silent:
            print("[ExplainerSmoothGRAD] Calculating gradients")
        gradient_sum = tf.zeros(shape=examples[0][0].shape)
        for ex in examples:
            gradient_sum += self._get_gradients(ex, do_sum=False)
        gradient_sum /= len(examples)
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
