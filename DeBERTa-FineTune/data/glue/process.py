from transformers import PretrainedConfig
from configs.glue.cfg import TASK_TO_KEYS


def preprocess_data(data, model, tokenizer, auto_config, num_labels, label_list, 
                    is_regression, logger, cfg, accelerator):
    # Preprocessing the datasets
    if cfg.DATA.TASK_NAME is not None:
        sentence1_key, sentence2_key = TASK_TO_KEYS[cfg.DATA.TASK_NAME]
    else:
        # Again, we try to have some nice defaults but don't hesitate to tweak to your use case.
        non_label_column_names = [name for name in data["train"].column_names if name != "label"]
        if "sentence1" in non_label_column_names and "sentence2" in non_label_column_names:
            sentence1_key, sentence2_key = "sentence1", "sentence2"
        else:
            if len(non_label_column_names) >= 2:
                sentence1_key, sentence2_key = non_label_column_names[:2]
            else:
                sentence1_key, sentence2_key = non_label_column_names[0], None

    # Some models have set the order of the labels to use, so let's make sure we do use it.
    label_to_id = None
    if (
        model.config.label2id != PretrainedConfig(num_labels=num_labels).label2id
        and cfg.DATA.TASK_NAME is not None
        and not is_regression
    ):
        # Some have all caps in their config, some don't.
        label_name_to_id = {k.lower(): v for k, v in model.config.label2id.items()}
        if list(sorted(label_name_to_id.keys())) == list(sorted(label_list)):
            logger.info(
                f"The configuration of the model provided the following label correspondence: {label_name_to_id}. "
                "Using it!"
            )
            label_to_id = {i: label_name_to_id[label_list[i]] for i in range(num_labels)}
        else:
            logger.warning(
                "Your model seems to have been trained with labels, but they don't match the dataset: ",
                f"model labels: {list(sorted(label_name_to_id.keys()))}, dataset labels: {list(sorted(label_list))}."
                "\nIgnoring the model labels as a result.",
            )
    elif cfg.DATA.TASK_NAME is None:
        label_to_id = {v: i for i, v in enumerate(label_list)}
    else:
        pass

    if label_to_id is not None:
        model.config.label2id = label_to_id
        model.config.id2label = {id: label for label, id in auto_config.label2id.items()}
    elif cfg.DATA.TASK_NAME is not None and not is_regression:
        model.config.label2id = {l: i for i, l in enumerate(label_list)}
        model.config.id2label = {id: label for label, id in auto_config.label2id.items()}

    def generate_features(examples):
        # Tokenize the texts
        texts = (
            (examples[sentence1_key],) if sentence2_key is None else (examples[sentence1_key], examples[sentence2_key])
        )
        result = tokenizer(*texts, padding="max_length" if cfg.DATA.PAD_TO_MAX_SEQ_LENGTH else False, max_length=cfg.DATA.MAX_SEQ_LENGTH, truncation=True)

        if "label" in examples:
            if label_to_id is not None:
                # Map labels to IDs (not necessary for GLUE tasks)
                # This situation may occur: some value in 'examples["label"]' does not existed in 'label_to_id',
                # etc. for MNLI testing set, all of examples["label"] is -1
                result["labels"] = [label_to_id.get(l) for l in examples["label"]]
                for i, label in enumerate(result["labels"]):
                    if label is None:
                        logger.warning(f"=> label {examples['label'][i]} of example{i} "
                                       f"does not in range(0, {num_labels}), please pay attention!")
            else:
                # In all cases, rename the column to labels because the model will expect that.
                result["labels"] = examples["label"]

        return result

    with accelerator.main_process_first():
        features = data.map(
            generate_features,
            batched=True,
            remove_columns=data["train"].column_names,
            desc="Running tokenizer on dataset",
            load_from_cache_file=cfg.DATA.LOAD_FROM_CACHE
        )
    
    return features
