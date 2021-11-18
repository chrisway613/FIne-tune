from datasets import load_dataset, load_metric


def load_data(name: str):
    """Load dataset and display its structure."""

    # Dataset will be downloaded & prepared to: ~/.cache/huggingface/datasets/squad/plain_text/1.0.0
    dataset = load_dataset(name, keep_in_memory=True)
    # for k, v in dataset.items():
    #     print(f'{k}: {v}')

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

    return dataset, load_metric(name)
