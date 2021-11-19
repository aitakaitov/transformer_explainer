import tensorflow as tf
import numpy as np
from transformer_explainer.utils import utils
from transformer_explainer.utils.explanation import Explanation


class ExplainerIG:
    def __init__(self, model, embeddings_tensor, baseline_tolerance=0.05):
        self.model = model
        self.embeddings = embeddings_tensor
        self.baseline_tolerance = baseline_tolerance

        self.embeddings_max = float(tf.reduce_max(embeddings_tensor))
        self.embeddings_min = float(tf.reduce_min(embeddings_tensor))

        self.input_classification = 0
        self.input_embeds = None
        self.x_input = False
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
                tokens: list, interpolation_steps=50, baseline=None, x_input=False, silent=True):

        self.input_embeds = input_embeds
        self.x_input = x_input
        self.silent = silent

        if not silent:
            print("[ExplainerIG] Classifying input")

        self.input_classification = self._classify((input_embeds, attention_mask, token_type_ids))

        if baseline is None:
            baseline = self._generate_baseline(input_embeds, attention_mask, token_type_ids)

        interpolated_examples = self._generate_examples(baseline, input_embeds, attention_mask, token_type_ids,
                                                        interpolation_steps)

        interpolated_gradients = self._get_interpolated_gradients(interpolated_examples)
        integrated_gradients = self._compute_ig(input_embeds, baseline, interpolated_gradients)

        if not silent:
            print("[ExplainerIG] Attribution sum: " + str(float(tf.reduce_sum(integrated_gradients))))

        return Explanation(tokens, integrated_gradients.numpy().tolist(), self.input_classification)

    def _compute_ig(self, input_embeds, baseline, gradients):
        if not self.silent:
            print("[ExplainerIG] Integrating computed gradients")
        gradients = tf.convert_to_tensor(gradients.numpy().astype(np.float32))
        gradients = (gradients[:-1] + gradients[1:]) / 2.0
        avg_gradients = tf.reduce_mean(gradients, axis=0)
        integrated_gradients = (tf.convert_to_tensor(input_embeds.astype(np.float32)) - tf.cast(baseline, dtype=tf.float32)) * avg_gradients

        if self.x_input:
            integrated_gradients = integrated_gradients * self.input_embeds

        integrated_gradients = tf.reduce_sum(integrated_gradients, axis=1)

        return integrated_gradients

    def _get_interpolated_gradients(self, interpolated_examples):
        if not self.silent:
            print("[ExplainerIG] Computing gradients for interpolated samples")
        interpolated_gradients = []
        for ex in interpolated_examples:
            grads = tf.squeeze(self._get_gradients(ex), axis=0)
            interpolated_gradients.append(grads)

        return tf.convert_to_tensor(interpolated_gradients)

    def _generate_examples(self, baseline, input_embeds, attention_mask, token_type_ids, steps):
        if not self.silent:
            print("[ExplainerIG] Linear interpolation in " + str(steps) + " steps")
        interpolated_examples = []
        for step in range(steps + 1):
            example = baseline + (step / steps) * (input_embeds - baseline)
            interpolated_examples.append((example, attention_mask, token_type_ids))

        return interpolated_examples

    def _generate_baseline(self, input_embeds: np.ndarray, attention_mask: np.ndarray, token_type_ids: np.ndarray):
        if not self.silent:
            print("[ExplainerIG] Generating baseline")
        zeros = np.zeros(shape=input_embeds.shape)
        res = self._classify((zeros, attention_mask, token_type_ids))

        if not self.silent:
            print("[ExplainerIG] All zero baseline classification result: " + str(res))
        if abs(res - 0.5) <= self.baseline_tolerance:
            return zeros
        else:
            if not self.silent:
                print("[ExplainerIG] Creating baseline trough gradients")
            token_count = attention_mask.tolist().index(0) + 1
            baseline = np.random.uniform(low=self.embeddings_min, high=self.embeddings_max, size=(token_count, input_embeds.shape[1]))
            baseline = np.concatenate((baseline, np.zeros(shape=(input_embeds.shape[0] - token_count, input_embeds.shape[1]))), axis=0)

            lr = 0.5
            while abs(res - 0.5) > self.baseline_tolerance:
                res = self._classify((baseline, attention_mask, token_type_ids))
                grads = self._get_gradients((baseline, attention_mask, token_type_ids))
                grads = tf.squeeze(grads, axis=0)
                if not self.silent:
                    print("[ExplainerIG] Baseline classification: " + str(res))
                if res > 0.5:
                    baseline = lr * (-1 * grads) + baseline
                else:
                    baseline = lr * grads + baseline
                lr = lr / 2

        if not self.silent:
            print("[ExplainerIG] Final random baseline classification result: " + str(res))
        return baseline
