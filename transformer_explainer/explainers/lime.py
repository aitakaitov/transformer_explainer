import numpy as np
from scipy.spatial.distance import cosine
from transformer_explainer.utils import utils
from transformer_explainer.utils.explanation import Explanation
from sklearn.linear_model import LinearRegression


class ExplainerLIME:
    def __init__(self, model, pad_token_embedding):
        self.model = model
        self.pad_token_embedding = pad_token_embedding.numpy()
        self.input_classification = 0
        self.silent = True

    def _classify(self, example):
        return float(self.model(utils.to_tensor_and_expand(example)))

    def explain(self, input_embeds: np.ndarray, attention_mask: np.ndarray, token_type_ids: np.ndarray,
                tokens: list, pertubation_method='tokens', sample_count=100, noise_size=0.01, cover_prob=0.125,
                silent=True):
        self.silent = silent
        if not self.silent:
            print("[ExplainerLIME] Classifying input")
        self.input_classification = self._classify((input_embeds, attention_mask, token_type_ids))

        if pertubation_method == 'tokens':
            examples = self._generate_examples_covering(input_embeds, attention_mask, token_type_ids, sample_count,
                                                        cover_prob)
        elif pertubation_method == 'noise':
            examples = self._generate_examples_noise(input_embeds, attention_mask, token_type_ids, sample_count,
                                                     noise_size)

        X, y = self._get_input_output_pairs(examples)

        if np.array_equal(X[0], X[1]):
            print()

        w = self._weigh_examples(X, input_embeds)

        linreg = LinearRegression().fit(X, y, w)
        weights = linreg.coef_
        weights = np.reshape(weights, input_embeds.shape)

        weights = np.sum(weights, axis=1)
        return Explanation(tokens=tokens, attributions=weights, classification=self.input_classification)

    def _matrix_similarity(self, A, B):
        return cosine(A, B)

    def _weigh_examples(self, X, input_embeds):
        if not self.silent:
            print("[ExplainerLIME] Weighing examples")
        weights = []
        input_embeds = input_embeds.flatten()
        for ex in X:
            weights.append(1 / self._matrix_similarity(ex, input_embeds))

        return weights

    def _get_input_output_pairs(self, examples):
        if not self.silent:
            print("[ExplainerLIME] Classifying samples")
        X = []
        y = []
        for example in examples:
            X.append(example[0].flatten())
            y.append(self._classify(example))

        return X, y

    def _generate_examples_covering(self, input_embeds, attention_mask, token_type_ids, steps, cover_prob):
        if not self.silent:
            print("[ExplainerLIME] Generating " + str(steps) + " samples trough token covering")
        examples = []
        mask_values = [0, 1]
        except_indices = [0, 511, 512, 1023, 1024, 1535]
        mask_weights = [cover_prob, 1-cover_prob]

        for i in range(steps):
            example = np.zeros(shape=input_embeds.shape)
            for j in range(input_embeds.shape[0]):
                if j in except_indices:
                    continue
                dec = int(np.random.choice(mask_values, p=mask_weights))
                if dec == 0:
                    example[j] = self.pad_token_embedding
                else:
                    example[j] = input_embeds[j]

            examples.append((example, attention_mask, token_type_ids))

        return examples

    def _generate_examples_noise(self, input_embeds, attention_mask, token_type_ids, steps, noise_size):
        if not self.silent:
            print("[ExplainerLIME] Generating " + str(steps) + " samples trough uniform noise")
        stdev = noise_size * (np.max(input_embeds) - np.min(input_embeds))
        noisy = []
        for i in range(steps):
            noise = np.random.normal(0, stdev, input_embeds.shape).astype(np.float32)
            noisy.append((input_embeds + noise, attention_mask, token_type_ids))

        return noisy