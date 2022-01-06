from datasets import load_dataset, load_metric


def load_data(name="glue", task_name=None, train_file=None, val_file=None):
    """
    Get the datasets: you can either provide your own CSV/JSON training and evaluation files (see below)
    or specify a GLUE benchmark task (the dataset will be downloaded automatically from the datasets Hub).
    
    In distributed training, the load_dataset function guarantee that 
    only one local process can concurrently download the dataset.
    """

    # Downloading and loading a dataset from the hub.
    if task_name is not None:
        data = load_dataset(name, task_name)
    # Loading the dataset from local csv or json file.
    else:
        # For CSV/JSON files, this script will use as labels the column called 'label' and as pair of sentences the
        # sentences in columns called 'sentence1' and 'sentence2' if such column exists or the first two columns not named
        # label if at least two columns are provided.

        # If the CSVs/JSONs contain only one non-label column, the script does single sentence classification on this
        # single column. You can easily tweak this behavior (see below)
        data_files = {}
        if train_file is not None:
            data_files["train"] = train_file
        if val_file is not None:
            data_files["validation"] = val_file

        extension = (train_file if train_file is not None else val_file).split(".")[-1]
        data = load_dataset(extension, data_files=data_files)
    # See more about loading any type of standard or custom dataset at
    # https://huggingface.co/docs/datasets/loading_datasets.html.

    # Labels
    if task_name is not None:
        try:
            metric_computor = load_metric("glue", task_name)
        except:
            metric_computor = None

        is_regression = task_name == "stsb"
        if is_regression:
            label_list = []
            num_labels = 1
        else:
            label_list = data["train"].features["label"].names
            num_labels = len(label_list)
    else:
        # Trying to have good defaults here, don't hesitate to tweak to your needs.
        is_regression = data["train"].features["label"].dtype in ["float32", "float64"]
        if is_regression:
            label_list = []
            num_labels = 1
        else:
            try:
                metric_computor = load_metric("accuracy")
            except:
                metric_computor = None
                
            # A useful fast method:
            # https://huggingface.co/docs/datasets/package_reference/main_classes.html#datasets.Dataset.unique
            label_list = data["train"].unique("label")
            label_list.sort()  # Let's sort it for determinism
            num_labels = len(label_list)
    
    return data, label_list, num_labels, is_regression, metric_computor
