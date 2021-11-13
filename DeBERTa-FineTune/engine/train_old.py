import os
import sys
import torch

from tqdm.auto import tqdm
from collections import defaultdict, OrderedDict

from datasets import load_dataset, load_metric
from transformers import AutoTokenizer, AutoModelForQuestionAnswering, AdamW, get_scheduler

from torch.optim import Adam
from torch.utils.data.dataloader import DataLoader

from config import *


# Current absolute path
BASE_DIR = os.path.dirname(__file__)
# Add parent directory to system path
sys.path.append(os.path.join(BASE_DIR, '..'))


def load_data(path_or_name: str):
    """Load dataset and display its structure."""

    print(f'Loading "{path_or_name}" dataset..')
    dataset = load_dataset(path_or_name, keep_in_memory=True)
    for k, v in dataset.items():
        print(f'{k}: {v}')

    '''
    [SQUADv1.1]
        train: Dataset({
            features: ['id', 'title', 'context', 'question', 'answers'],
            num_rows: 87599
        })
        validation: Dataset({
            features: ['id', 'title', 'context', 'question', 'answers'],
            num_rows: 10570
        })
    '''

    return dataset


def generate_train_features(examples):
    """Generate input features for training set, 
    the main work is tokenizing & label each example(span)"""

    # i.   Filter whitespace in questions(which will make the truncation of the context fail)
    examples['question'] = [q.lstrip() for q in examples['question']]

    # ii.  Tokenize examples
    tokenizer = AutoTokenizer.from_pretrained(MODEL_CHECKPOINT)
    tokenized_examples = tokenizer(
        examples['question' if PAD_ON_RIGHT else 'context'],
        examples['context' if PAD_ON_RIGHT else 'question'],
        truncation='only_second' if PAD_ON_RIGHT else 'only_first',
        max_length=MAX_LENGTH, stride=DOC_STRIDE, padding='max_length',
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
            while seq_ids_per_example[start_token_index] != (1 if PAD_ON_RIGHT else 0):
                start_token_index += 1
            # End token index of the current span in the text
            end_token_index = len(input_ids_per_example) - 1
            while seq_ids_per_example[end_token_index] != (1 if PAD_ON_RIGHT else 0):
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
    tokenizer = AutoTokenizer.from_pretrained(MODEL_CHECKPOINT)
    tokenized_examples = tokenizer(
        examples['question' if PAD_ON_RIGHT else 'context'],
        examples['context' if PAD_ON_RIGHT else 'question'],
        truncation='only_second' if PAD_ON_RIGHT else 'only_first',
        max_length=MAX_LENGTH, stride=DOC_STRIDE, padding='max_length',
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
        context_index = 1 if PAD_ON_RIGHT else 0
        seq_ids = tokenized_examples.sequence_ids(i)

        # Set to None the offset_mapping that are not part of the context so it's easy to determine if a token
        # position is part of the context or not.
        tokenized_examples['offset_mapping'][i] = [
            offset if seq_ids[j] == context_index else None
            for j, offset in enumerate(tokenized_examples['offset_mapping'][i])
        ]

    return tokenized_examples


def generate_features(dataset='train'):
    """Tokenize the original input texts"""
    return generate_train_features if dataset == 'train' else generate_val_features


def postprocess_predictions(examples, features, raw_predictions,
                            n_best_size=20, max_answer_length=30):
    """Postprocess the raw predictions, in order to compute metric."""
    tokenizer = AutoTokenizer.from_pretrained(MODEL_CHECKPOINT)

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
            start_token_indices = torch.argsort(start_logits, descending=True)[:n_best_size]
            end_token_indices = torch.argsort(end_logits, descending=True)[:n_best_size]

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
                            end_token_index - start_token_index + 1 > max_answer_length:
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

        if SQUAD_VER == 1:
            predictions[example['id']] = best_answer['text']
        else:
            predictions[example['id']] = best_answer['text'] \
                if best_answer['score'] > min_null_score else ''

    # Prediction format different from SQUAD v1 & v2
    if SQUAD_VER == 1:
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


if __name__ == '__main__':
    '''Load data'''
    data = load_data(DATASET_NAME)
    train_set, val_set = data['train'], data['validation']
    # train_questions, train_answers = train_set['question'], train_set['answers']
    # for q, a in zip(train_questions[:10], train_answers[:10]):
    #     print(f'Question: {q}; Answer: {a["text"][0]}')

    # sample = train_set[0]
    # print(sample)

    '''Preprocess data'''
    # example = train_set[0]
    # tokenizer = AutoTokenizer.from_pretrained(MODEL_CHECKPOINT)
    # We need to account for the special case where the model expects padding on the left
    # (in which case we switch the order of the question and the context)
    # pad_on_right = tokenizer.padding_side == 'right'
    # tokenized_example = tokenizer(
    #     example["question" if pad_on_right else "context"],
    #     example["context" if pad_on_right else "question"],
    #     truncation="only_second" if pad_on_right else "only_first",
    #     max_length=MAX_LENGTH, stride=DOC_STRIDE, padding=MAX_LENGTH,
    #     return_offsets_mapping=True,
    #     return_overflowing_tokens=True
    # )
    # print(tokenized_example.keys(), '\n')
    # for input_id in tokenized_example['input_ids']:
    #     print(tokenizer.decode(input_id))

    # first_token = tokenizer.decode(tokenized_example['input_ids'][0][1])
    # first_token_start, first_token_end = tokenized_example['offset_mapping'][0][1]
    # first_word = example['question'][first_token_start:first_token_end]
    # print(first_token, first_word)

    # Apply all sentences in training set
    old_columns = train_set.column_names
    # train_set, val_set = train_set.select(range(96)), val_set.select(range(48))
    # 'batched=True' means to use multi-threads
    tokenized_train_data = train_set.map(generate_features(),
                                         batched=True, remove_columns=old_columns)
    tokenized_val_data = val_set.map(generate_features(),
                                     batched=True, remove_columns=old_columns)
    # print(type(tokenized_train_data))
    # tokenized_sample = tokenized_train_data[3]
    # print(type(tokenized_sample), tokenized_sample.keys())
    # print(tokenized_sample)
    # tokenized_sample = generate_train_features(train_set[3:4])
    # print(tokenized_sample)
    # start_pos, end_pos = tokenized_sample['start_positions'][0], tokenized_sample['end_positions'][0]
    # ans_token_ids = tokenized_sample['input_ids'][0][start_positions:(end_positions + 1)]
    # decoded_ans = tokenizer.decode(ans_token_ids)
    # ans = train_set[3]['answers']['text'][0]
    # print(f'decoded:{decoded_ans} original:{ans}')

    tokenized_train_data.set_format(DATA_FORMAT)
    train_data = tokenized_train_data.shuffle(SEED)

    tokenized_val_data.set_format(DATA_FORMAT)
    val_data = tokenized_val_data.shuffle(SEED)

    train_dataloader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
    val_dataloader = DataLoader(val_data, batch_size=BATCH_SIZE)
    print(f'There are {len(train_dataloader)} train batches total.')
    print(f'There are {len(val_dataloader)} val batches total.\n')

    '''Build model'''
    device = 'cpu'
    # device = torch.device('cuda:7') if torch.cuda.is_available() else 'cpu'

    model_name = MODEL_CHECKPOINT.split('/')[-1]
    if RESUME:
        history = os.path.join(OUT_DIR, f'{model_name}-{DATASET_NAME}')
        model = AutoModelForQuestionAnswering.from_pretrained(history)
    else:
        model = AutoModelForQuestionAnswering.from_pretrained(MODEL_CHECKPOINT)
    print(f'load model "{model_name}" finished!\n')
    model.to(device)

    '''Set optimizer & lr-scheduler'''
    # optimizer = AdamW(model.parameters(), lr=LR, eps=1e-6)
    optimizer = Adam(model.parameters(), lr=LR, eps=1e-6, weight_decay=1e-2)

    train_steps = EPOCHS * len(train_dataloader)
    lr_scheduler = get_scheduler(
        "linear",
        optimizer=optimizer,
        num_warmup_steps=WARMUP_STEPS,
        num_training_steps=train_steps
    )

    '''Train'''
    val_features = val_set.map(
        generate_features(dataset='val'),
        batched=True, remove_columns=val_set.column_names
    )
    val_features.set_format(type=val_features.format['type'],
                            columns=list(val_features.features.keys()))

    # for val_feat in val_features:
    #     print(val_feat)

    f1 = em = 0.
    metric = load_metric(DATASET_NAME)
    best_ckp_dir = os.path.join(OUT_DIR, f'{model_name}-{DATASET_NAME}-best')
    os.makedirs(best_ckp_dir, exist_ok=True)
    best_checkpoint = os.path.join(best_ckp_dir, 'pytorch_model.bin')

    progress_bar = tqdm(range(train_steps))
    for epoch in range(1, EPOCHS + 1):
        model.train()
        print(f'Train Epoch[{epoch}]')

        for i, batch in enumerate(train_dataloader):
            batch = {k: v.to(device) for k, v in batch.items()}
            # print(batch)
            outputs = model(**batch)
            # print(outputs)

            loss = outputs.loss
            loss.backward()
            print(f'Step{i} Train Loss: {loss.item():.5f}')

            optimizer.step()
            lr_scheduler.step()

            optimizer.zero_grad()
            progress_bar.update()

        # Save checkpoint after each training epoch
        checkpoint_dir = os.path.join(OUT_DIR, f'{model_name}-{DATASET_NAME}-{epoch}')
        os.makedirs(checkpoint_dir, exist_ok=True)
        # This will save 'config.json' & 'pytorch_model.bin'
        model.save_pretrained(checkpoint_dir)

        checkpoint = os.path.join(checkpoint_dir, 'pytorch_model.bin')
        print(f'checkpoint "{checkpoint}" saved.\n')

        print('-' * 50, '\n')

        '''Eval'''
        model.eval()
        print(f'Val Epoch[{epoch}]')

        with torch.no_grad():
            # Predicted logtis among all features
            start_logits, end_logits = [], []
            for j, batch in enumerate(val_dataloader):
                batch = {k: v.to(device) for k, v in batch.items()}
                outputs = model(**batch)

                loss = outputs.loss
                print(f'Step{j} Val Loss: {loss.item():.5f}')

                start_logits.append(outputs.start_logits)
                end_logits.append(outputs.end_logits)
        start_logits = torch.concat(start_logits)
        end_logits = torch.concat(end_logits)

        raw_predictions = [start_logits, end_logits]
        # Postprocess in order to compute metric
        predictions = postprocess_predictions(
            val_set, val_features, raw_predictions,
            n_best_size=N_BEST_ANSWERS, max_answer_length=MAX_ANSWER_LENGTH
        )
        references = [{'id': example['id'], 'answers': example['answers']} for example in val_set]
        # F1 & Exact Match
        val_result = metric.compute(predictions=predictions, references=references)

        val_str = 'Evaluation Results:\n'
        for k, v in val_result.items():
            val_str += f'{k}: {v:.5f} '
        print(val_str)

        if val_result['f1'] > f1 or val_result['exact_match'] > em:
            print('Gain best result: {val_result}')

            if val_result['f1'] > f1:
                f1 = val_result['f1']
                model.save_pretrained(best_ckp_dir)
                print(f'best checkpoint "{best_checkpoint}" saved')

            if val_result['exact_match'] > em:
                em = val_result['exact_match']

        print('-' * 50, '\n')

    print()
