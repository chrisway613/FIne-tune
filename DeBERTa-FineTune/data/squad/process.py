import torch

from transformers import AutoTokenizer
from collections import defaultdict, OrderedDict

import os
import sys

BASE_DIR = os.path.dirname(__file__)
sys.path.append(os.path.join(BASE_DIR, '..', '..'))

from configs.squad.cfg import _C


# Return a clone so that the defaults will not be altered
# This is for the "local variable" use pattern
config = _C.clone()


def generate_train_features(examples):
    """Generate input features for training set, 
    the main work is tokenizing & label each example(span)"""

    # i.   Filter whitespace in questions(which will make the truncation of the context fail)
    examples['question'] = [q.lstrip() for q in examples['question']]

    # ii.  Tokenize examples
    # Note: check if your model support the fast tokenizers
    tokenizer = AutoTokenizer.from_pretrained(config.MODEL.TYPE, use_fast=True)
    tokenized_examples = tokenizer(
        examples['question' if config.DATA.PAD_ON_RIGHT else 'context'],
        examples['context' if config.DATA.PAD_ON_RIGHT else 'question'],
        truncation='only_second' if config.DATA.PAD_ON_RIGHT else 'only_first',
        max_length=config.DATA.MAX_SEQ_LENGTH, stride=config.DATA.DOC_STRIDE, padding='max_length',
        return_overflowing_tokens=True, return_offsets_mapping=True
    )
    # print(f'raw tokenized examples: {tokenized_examples}\n')

    # Token input ids in each example
    input_ids = tokenized_examples['input_ids']
    # Map from token to character position in the original context
    offsets_mapping = tokenized_examples.pop('offset_mapping')
    # Map from a feature to its corresponding example.
    # (one example may correspond to multiple features since they have long context)
    sample_mapping = tokenized_examples.pop('overflow_to_sample_mapping')

    assert len(offsets_mapping) == len(input_ids), "the length of \
            'offsets_mapping' & 'input_ids' must be equal"
    assert len(offsets_mapping) == len(sample_mapping), "the length of 'offsets_mapping' & \
            'sample_mapping' must be equal"

    # iii. Label each example by its start & end position in its 'input_ids'
    tokenized_examples['start_positions'] = []
    tokenized_examples['end_positions'] = []

    for i, (sample_index, offsets_per_example, input_ids_per_example) in enumerate(zip(
            sample_mapping, offsets_mapping, input_ids)):
        # We will label impossible answers with the index of the CLS token.
        cls_index = input_ids_per_example.index(tokenizer.cls_token_id)
        # print(f'cls token index in i-sample: {cls_index}')
        # One example can give several spans,
        # this is the index of the example containing this span of text.
        answer = examples["answers"][sample_index]
        # print(f'{i}-answer: {answer}')

        # If no answers are given, set the 'cls_index' as answer.
        if not answer["answer_start"]:
            tokenized_examples['start_positions'].append(cls_index)
            tokenized_examples['end_positions'].append(cls_index)
        else:
            # Start/end character index of the answer in the text.
            answer_start_index = answer["answer_start"][0]
            answer_end_index = answer_start_index + len(answer["text"][0])
            # print(f'{i}-answer start - end: {answer_start_index} - {answer_end_index}\n')

            # iv. Find start & end token index of the current span in the text
            # Grab the sequence corresponding to that example
            # (to know what is the context and what is the question)
            seq_ids_per_example = tokenized_examples.sequence_ids(i)
            assert len(seq_ids_per_example) == len(input_ids_per_example), "the length of \
                'seq_ids_per_example' & 'input_ids_per_example' must be equal"

            # Start token index of the current span in the text
            start_token_index = 0
            while seq_ids_per_example[start_token_index] != (1 if config.DATA.PAD_ON_RIGHT else 0):
                start_token_index += 1
            # End token index of the current span in the text
            end_token_index = len(input_ids_per_example) - 1
            while seq_ids_per_example[end_token_index] != (1 if config.DATA.PAD_ON_RIGHT else 0):
                end_token_index -= 1

            # Current span's start char index of the text
            start_char_index = offsets_per_example[start_token_index][0]
            # Current span's end char index of the text
            end_char_index = offsets_per_example[end_token_index][1]
            # Detect if the answer is out of the span
            # (in which case this feature will be labeled with the CLS index)
            if not (start_char_index <= answer_start_index <
                    answer_end_index <= end_char_index):
                tokenized_examples['start_positions'].append(cls_index)
                tokenized_examples['end_positions'].append(cls_index)
            # Otherwise move the token_start_index and token_end_index
            # to the two ends of the answer.
            else:
                while start_char_index <= answer_start_index and \
                        start_token_index < len(offsets_per_example) - 1:
                    start_token_index += 1
                    start_char_index = offsets_per_example[start_token_index][0]
                while end_char_index >= answer_end_index and end_token_index:
                    end_token_index -= 1
                    end_char_index = offsets_per_example[end_token_index][1]

                # Insert assertion for ensuring label correctness
                answer_input_ids = input_ids_per_example[
                                   (start_token_index - 1):(end_token_index + 1 + 1)]
                decoded_answer = tokenizer.decode(answer_input_ids)
                truely_answer = answer['text'][0]
                # assert truely_answer.strip() in decoded_answer.strip(), \
                #     f"\nthe answer '{truely_answer}' not in labeled '{decoded_answer}'\n"
                if truely_answer.strip() not in decoded_answer.strip():
                    print(f"\nWarning: the answer '{truely_answer}' \
                        not in labeled '{decoded_answer}'\n")

                tokenized_examples['start_positions'].append(start_token_index - 1)
                tokenized_examples['end_positions'].append(end_token_index + 1)

    return tokenized_examples


def generate_val_features(examples):
    """Generate input features for validation set,
    the main work is tokenize, record example id & filter our invalid offset mapping(that
    is, the token is not part of the context but the question)"""

    # i.   Filter whitespace in questions(which will make the truncation of the context fail)
    examples['question'] = [q.lstrip() for q in examples['question']]

    # ii.  Tokenize
    # Tokenize our examples with truncation and maybe padding, but keep the overflows using a stride. This results
    # in one example possible giving several features when a context is long, each of those features having a
    # context that overlaps a bit the context of the previous feature.
    # Note: check if your model support the fast tokenizers
    tokenizer = AutoTokenizer.from_pretrained(config.MODEL.TYPE, use_fast=True)
    tokenized_examples = tokenizer(
        examples['question' if config.DATA.PAD_ON_RIGHT else 'context'],
        examples['context' if config.DATA.PAD_ON_RIGHT else 'question'],
        truncation='only_second' if config.DATA.PAD_ON_RIGHT else 'only_first',
        max_length=config.DATA.MAX_SEQ_LENGTH, stride=config.DATA.DOC_STRIDE, padding='max_length',
        return_overflowing_tokens=True, return_offsets_mapping=True
    )

    # iii. Map feature to its example & filter out those tokens which is not part of context
    tokenized_examples['example_id'] = []
    # Since one example might give us several features if it has a long context, we need a map from a feature to
    # its corresponding example. This key gives us just that.
    sample_mapping = tokenized_examples.pop("overflow_to_sample_mapping")

    for i, sample_index in enumerate(sample_mapping):
        # One example can give several spans, 
        # this is the index of the example containing this span of text.
        tokenized_examples['example_id'].append(examples['id'][sample_index])

        # Grab the sequence corresponding to that example 
        # (to know what is the context and what is the question).
        context_index = 1 if config.DATA.PAD_ON_RIGHT else 0
        seq_ids = tokenized_examples.sequence_ids(i)

        # Set to None the offset_mapping that are not part of the context so it's easy to determine if a token
        # position is part of the context or not.
        tokenized_examples['offset_mapping'][i] = [
            offset if seq_ids[j] == context_index else None
            for j, offset in enumerate(tokenized_examples['offset_mapping'][i])
        ]

    return tokenized_examples


def generate_features(mode='train'):
    """Tokenize the original input texts"""
    return generate_train_features if mode == 'train' else generate_val_features


def postprocess_predictions(examples, features, raw_predictions, config):
    """Postprocess the raw predictions, in order to compute metric."""
    tokenizer = AutoTokenizer.from_pretrained(config.MODEL.TYPE)

    # i. Build a mapping relations from example to its features
    example_id_to_index = {example_id: i for i, example_id in enumerate(examples['id'])}
    features_per_example = defaultdict(list)
    for feature_index, feature in enumerate(features):
        example_index = example_id_to_index[feature['example_id']]
        features_per_example[example_index].append(feature_index)

    # Logging
    print(f"Post-processing {len(examples)} example predictions \
        split into {len(features)} features..")

    # ii. Gather the prediction results of all examples
    predictions = OrderedDict()
    # Logits of all features
    all_start_logits, all_end_logtis = raw_predictions

    # Loop over all examples
    for example_index, example in enumerate(examples):
        # Candidate answers of current example
        valid_answers = []
        # We have to predict the impossible answer 
        # when all the features give a high score to the impossible answer 
        # (since one feature could predict the impossible answer just because 
        # the answer isn't in the part of the context it has access too), 
        # which is why the score of the impossible answer for one example is 
        # the minimum of the scores for the impossible answer in each feature 
        # generated by the example.
        # Only used in SQUADv2
        min_null_score = None
        # Context of current example
        context = example['context']

        # Indices of the features associated to the current example
        feature_indices = features_per_example[example_index]
        # Looping through all the features associated to the current example
        for feature_index in feature_indices:
            feature = features[feature_index]
            # This is what will allow us to map the positions in our logits 
            # to span of texts in the original context.
            offset_mapping = feature["offset_mapping"]

            start_logits = all_start_logits[feature_index]
            end_logits = all_end_logtis[feature_index]

            # 2.1 Update minimum null prediction score
            cls_index = feature['input_ids'].index(tokenizer.cls_token_id)
            feature_null_score = start_logits[cls_index] + end_logits[cls_index]
            if min_null_score is None or feature_null_score > min_null_score:
                min_null_score = feature_null_score

            # 2.2 Go through all possibilities for the 
            # `n_best_size` greater start and end logits
            # These indices are correspond to ‘offset_mapping’
            start_token_indices = torch.argsort(start_logits, descending=True)[:config.DATA.N_BEST_ANSWERS]
            end_token_indices = torch.argsort(end_logits, descending=True)[:config.DATA.N_BEST_ANSWERS]

            for start_token_index in start_token_indices:
                for end_token_index in end_token_indices:
                    # Ignore invalid ones
                    if start_token_index >= len(offset_mapping) or \
                            end_token_index >= len(offset_mapping):
                        continue
                    if offset_mapping[start_token_index] is None or \
                            offset_mapping[end_token_index] is None:
                        continue
                    if start_token_index > end_token_index or \
                            end_token_index - start_token_index + 1 > config.DATA.MAX_ANSWER_LENGTH:
                        continue

                    # Start char index in context
                    start_char_ctx_index = offset_mapping[start_token_index][0]
                    # End char index in context
                    end_char_ctx_index = offset_mapping[end_token_index][1]

                    predicted_answer = {
                        'text': context[start_char_ctx_index:(end_char_ctx_index + 1)],
                        'score': start_logits[start_token_index] + end_logits[end_token_index]
                    }
                    valid_answers.append(predicted_answer)

        # 2.3 Grab the best answer for each example & map example id to the answer text
        best_answer = sorted(valid_answers, key=lambda ans: ans['score'], reverse=True)[0] \
            if valid_answers else {'text': '', 'score': 0.}

        if config.DATA.DATASET == 'squad':
            predictions[example['id']] = best_answer['text']
        else:
            predictions[example['id']] = best_answer['text'] \
                if best_answer['score'] > min_null_score else ''

    # Prediction format different from SQUAD v1 & v2
    if config.DATA.DATASET == 'squad':
        predictions = [
            {'id': example_id, 'prediction_text': text}
            for example_id, text in predictions.items()
        ]
    else:
        predictions = [
            {'id': example_id, 'prediction_text': text, 'no_answer_probability': 0.}
            for example_id, text in predictions.items()
        ]

    return predictions
