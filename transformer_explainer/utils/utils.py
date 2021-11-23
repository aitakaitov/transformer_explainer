import tensorflow as tf


def to_tensor_and_expand(example):
    example = (
        tf.expand_dims(tf.convert_to_tensor(example[0], name='inputs_embeds'), axis=0),
        tf.expand_dims(tf.convert_to_tensor(example[1], name='attention_mask'), axis=0),
        tf.expand_dims(tf.convert_to_tensor(example[2], name='token_type_ids'), axis=0)
    )
    return example


def get_html_header():
    return '<html lang="cs"><head><meta http-equiv="Content-Type" content="text/html; charset=UTF-8"></head>' \
           '<body style="font-size: 28px;font-family:Arial">'


def get_html_footer():
    return '</body></html>'


def get_html_classification_header(classification_value):
    if classification_value > 0.5:
        text = "Komerční článek"
    else:
        text = "Nekomerční článek"

    return '<div style="width:100%;text-align:center;font-size: 25px;padding-bottom:10px"><strong>' \
           'Predikce sítě: {:0.4f}'.format(classification_value) + ' -> ' + text + '</strong></div>'


def get_word_html(text, val, was_tag_start):
    # we dont color the paragraphs
    if text == "<":
        # word = "&lt"
        was_tag_start = True
        return "<", was_tag_start
    elif text == ">":
        # word = "&gt"
        was_tag_start = False
        return ">", was_tag_start
    if was_tag_start:
        return text, was_tag_start
    if val < 0:
        html = '<span style="color:rgb(' + str(128 - int(64 * val)) + ", " + str(128 - int(64 * val)) + \
               "," + str(int(128 * val + 127)) + ');">'
    else:
        html = '<span style="color:rgb(' + str(int(128 * val + 127)) + ", " + str(128 - int(64 * val)) + "," + str(
                128 - int(64 * val)) + ');">'
    if text not in ["!", "?", ".", ",", '"']:
        html = " " + html + text
    else:
        html = html + text
    html += "</span>"
    return html, was_tag_start


def keep_top_k_abs(values, k):
    non_zero_count = 0
    for val in values:
        if val != 0:
            non_zero_count += 1

    if non_zero_count < k:
        return values

    a = non_zero_count - k
    for i in range(non_zero_count - k):
        min_val = float('inf')
        min_idx = -1
        for j in range(len(values)):
            if values[j] != 0:
                if abs(values[j]) < min_val:
                    min_val = abs(values[j])
                    min_idx = j

        values[min_idx] = 0
    return values


def remove_positive(attributions):
    for i in range(len(attributions)):
        if attributions[i] > 0:
            attributions[i] = 0.0

    return attributions


def remove_negative(attributions):
    for i in range(len(attributions)):
        if attributions[i] < 0:
            attributions[i] = 0.0

    return attributions


def get_percentile_value_abs(values, percentile):
    abs_vals = []
    for attr in values:
        abs_vals.append(abs(attr))

    abs_vals.sort()
    perc_value = abs_vals[int(percentile * (len(values) - 1))]

    return perc_value


def remove_percentile_abs(attributions, percentile: float):
    perc_value = get_percentile_value_abs(attributions, percentile)

    for i in range(len(attributions)):
        if abs(attributions[i]) > perc_value:
            attributions[i] = 0.0

    return attributions


def scale_attributions(attributions):
    _max = -1
    for attr in attributions:
        if abs(attr) > _max:
            _max = abs(attr)

    for i in range(len(attributions)):
        attributions[i] = attributions[i] / _max

    return attributions


def get_word_list_sum(tokens, attributions):
    words = []
    word_values = []

    value = 0
    word = ""
    special_tokens = ["[CLS]", "[PAD]", "[SEP]"]
    for i in range(0, min(len(tokens), 1536)):
        if tokens[i] in special_tokens:
            if tokens[i] == "[PAD]":
                word = ""
                break
            if word == "":
                continue
            else:
                words.append(word)
                word_values.append(value)
        elif "##" in tokens[i]:
            word += tokens[i][2:]
            value += attributions[i]
        else:
            if word == "":
                word = tokens[i]
                value = attributions[i]
            else:
                words.append(word)
                word_values.append(value)
                word = tokens[i]
                value = attributions[i]

    if word != "":
        words.append(word)
        word_values.append(value)
    return words, word_values


def get_word_list_avg(tokens, attributions):
    words = []
    word_values = []

    count = 0
    value = 0
    word = ""
    special_tokens = ["[CLS]", "[PAD]", "[SEP]"]
    for i in range(0, min(len(tokens), 1536)):
        if tokens[i] in special_tokens:
            if tokens[i] == "[PAD]":
                word = ""
                break
            if word == "":
                continue
            else:
                words.append(word)
                word_values.append(value / count)
                count = 0
        elif "##" in tokens[i]:
            word += tokens[i][2:]
            value += attributions[i]
            count += 1
        else:
            if word == "":
                word = tokens[i]
                value = attributions[i]
                count = 1
            else:
                words.append(word)
                word_values.append(value / count)
                word = tokens[i]
                value = attributions[i]
                count = 1

    if word != "":
        words.append(word)
        word_values.append(value / count)
    return words, word_values


def tokenize(tokenizer, plaintext):
    tokens = tokenizer.tokenize(plaintext, add_special_tokens=True)
    if 1024 < len(tokens):
        block_num = 3
    elif 512 < len(tokens) <= 1024:
        block_num = 2
    elif len(tokens) <= 512:
        block_num = 1

    cls = 2
    sep = 3

    if block_num == 1:
        x = tokenizer(plaintext, max_length=1536, truncation=True, padding='max_length')
    elif block_num == 2:
        x = tokenizer(plaintext, max_length=1534, truncation=True, padding='max_length')
        x.data['input_ids'] = x.data['input_ids'][0:511] + [sep, cls] + x.data['input_ids'][511:]
        x.data['attention_mask'] = [1, 1] + x.data['attention_mask']
        x.data['token_type_ids'] = [0, 0] + x.data['token_type_ids']
    elif block_num == 3:
        x = tokenizer(plaintext, max_length=1532, truncation=True, padding='max_length')
        temp = []
        temp = x.data['input_ids'][0:511] + [sep]
        temp += [cls] + x.data['input_ids'][511:1022] + [sep]
        temp += [cls] + x.data['input_ids'][1022:]
        x.data['input_ids'] = temp
        x.data['attention_mask'] = [1, 1, 1, 1] + x.data['attention_mask']
        x.data['token_type_ids'] = [0, 0, 0, 0] + x.data['token_type_ids']

    return x
