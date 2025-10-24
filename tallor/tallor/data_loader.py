import torch                                  # Core PyTorch library for tensors and autograd
import torch.utils.data as data              # Utilities for data loading (Dataset, DataLoader)
import os                                     # OS utilities for file paths
import random                                 # Random for sampling indices
import json                                   # JSON for reading data files
from tallor.utils import DataPoint, list_to_dict  # Custom utilities: DataPoint wraps examples; list_to_dict merges list of dicts
from copy import deepcopy                     # Deep copy to duplicate data without side effects
import spacy                                  # spaCy library for NLP parsing
from spacy.tokens import Doc                  # spaCy’s Doc object for token sequences

class Parser:
    """Custom spaCy parser that preserves pre-tokenization."""

    def __init__(self):
        self.parser = spacy.load('en_core_web_sm')  # Load English model
        self.parser.remove_pipe('ner')              # Remove spaCy’s built-in NER
        self.parser.tokenizer = self.custom_tokenizer  # Override tokenizer

    def custom_tokenizer(self, text):
        tokens = text.split(' ')                    # Split text on spaces only
        return Doc(self.parser.vocab, tokens)       # Build spaCy Doc from pre-split tokens

    def parse(self, sentence):
        return self.parser(sentence)                # Apply parser (POS, dependencies) to token sequence

class MissingDict(dict):
    """
    Dict that returns a default for missing keys without storing them.
    """

    def __init__(self, missing_val, generator=None) -> None:
        if generator:
            super().__init__(generator)            # Initialize with provided key→value pairs
        else:
            super().__init__()                     # Empty dict otherwise
        self._missing_val = missing_val             # Value to return for missing keys

    def __missing__(self, key):
        return self._missing_val                    # Return default on missing lookup

def format_label_fields(ner):
    """
    Convert list of (start, end, label) tuples into a MissingDict mapping spans to labels.
    """

    ner_dict = MissingDict(
        "",                                      # Default label = empty string
        (
            ((span_start, span_end), named_entity)
            for (span_start, span_end, named_entity) in ner  # Generator of key→value pairs
        )
    )
    return ner_dict                               # Return span→entity map

class DataSet(data.Dataset):
    """
    PyTorch Dataset for semi-supervised span-based NER.
    Splits data into labeled/unlabeled, enumerates spans, and prepares examples.
    """

    def __init__(self, root, filename, encoder, batch_size, ner_label, is_train, opt, mode='training'):
        self.batch_size = batch_size               # Batch size
        self.max_span_width = opt.max_span_width   # Max span length to consider
        self.ner_label = ner_label                 # Label vocabulary (maps labels↔ids)
        self.encoder = encoder                     # Tokenizer/encoder (e.g., BERT)
        self.parser = Parser()                     # Use custom spaCy parser

        self.label_data = []                       # Store true-labeled DataPoints
        self.unlabel_data = []                     # Store unlabeled DataPoints

        if mode == 'training':
            labeled_ratio = opt.labeled_ratio      # Fraction to keep labeled
            path = os.path.join(root, filename + ".json")  # JSON file path

            data = []
            with open(path, "r") as f:
                lines = f.readlines()              # Read all lines
                for line in lines:
                    data.append(json.loads(line))  # Parse JSON per line

            print(f'Begin processing {filename} dataset...')

            processed_data = self.preprocess(data)  # Convert raw JSON to (sentence, spans, labels)

            data = []
            for line in processed_data:
                sentence, spans, ner_labels = line
                data.append(DataPoint(
                    sentence=sentence,             # List of tokens
                    spans=spans,                   # All candidate spans
                    ner_labels=ner_labels,         # Gold label ids per span
                    parsed_tokens=self.parser.parse(' '.join(sentence)),  # spaCy parse
                    label_num=self.ner_label.get_num()  # Number of entity classes
                ))

            # Determine how many to label vs. unlabel
            if not is_train or labeled_ratio == 1:
                self.training_data = data        # All data labeled
                labeled_num = len(data)
                unlabeled_num = 0
            else:
                index = list(range(len(data)))
                labeled_index = index[:int(labeled_ratio * len(index))]  # First segment
                unlabel_index = index[int(labeled_ratio * len(index)):]

                self.label_data = [data[i] for i in labeled_index]      # Keep true-labeled

                for i in unlabel_index:
                    data[i].unlabel_reset()      # Mask out labels for semi-supervised
                    self.unlabel_data.append(data[i])

                self.training_data = deepcopy(self.label_data)  # Start train set from labeled only
                labeled_num = len(self.label_data)
                unlabeled_num = len(self.unlabel_data)

            print(f'Done. {filename} dataset has {len(data)} instances. '
                  f'\n Among them, we use {labeled_num} instances as labeled data, {unlabeled_num} instances as unlabeled data')
        else:  # Serving / inference mode
            data = self.read_and_process_unlabel_set(root, filename)
            self.unlabel_data = data
            self.training_data = []              # No supervised data
            print(f'We get {len(self.unlabel_data)} sentences.')

    def read_and_process_unlabel_set(self, root, filename):
        """
        Read unlabeled JSONL and preprocess for inference.
        """
        path = os.path.join(root, filename)
        data = []
        with open(path, "r") as f:
            lines = f.readlines()
            for line in lines:
                data.append(json.loads(line)['sentence'])  # Only sentences

        processed_data = self.preprocess(data, mode='serving')
        data = []
        for line in processed_data:
            sentence, spans = line
            data_point = DataPoint(
                sentence=sentence,
                spans=spans,
                ner_labels=[-1] * len(spans),    # Dummy labels
                parsed_tokens=self.parser.parse(' '.join(sentence)),
                label_num=self.ner_label.get_num()
            )
            data_point.unlabel_reset()           # Mask labels
            data.append(data_point)
        return data

    def preprocess(self, data, mode='training'):
        """
        Convert raw JSON or text into (sentence, spans, labels) tuples.
        """
        processed = []
        if mode == 'training':
            for line in data:
                for sentence, ner in zip(line["sentences"], line["ner"]):
                    ner_dict = format_label_fields(ner)  # Map spans→labels
                    sentence, spans, ner_labels = self.text_to_instance(sentence, ner_dict)
                    processed.append([sentence, spans, ner_labels])
        else:  # mode='serving'
            for sentence in data:
                sentence, spans = self.text_to_instance(sentence, None, mode=mode)
                processed.append([sentence, spans])
        return processed

    def text_to_instance(self, sentence, ner_dict, mode='training'):
        """
        Enumerate all spans and assign labels (or none in serving).
        """
        spans = []
        ner_labels = []
        if mode == 'training':
            for start, end in self.enumerate_spans(sentence, max_span_width=self.max_span_width):
                spans.append((start, end))
                ner_label = ner_dict[(start, end)]            # Lookup label or ""
                ner_labels.append(self.ner_label.get_id(ner_label))  # Convert to id
            return sentence, spans, ner_labels
        else:
            for start, end in self.enumerate_spans(sentence, max_span_width=self.max_span_width):
                spans.append((start, end))
            return sentence, spans

    def enumerate_spans(self, sentence, max_span_width, min_span_width=1):
        """
        Generate all contiguous spans up to max_span_width.
        """
        max_span_width = max_span_width or len(sentence)
        spans = []
        for start_index in range(len(sentence)):
            first_end = min(start_index + min_span_width - 1, len(sentence))
            last_end = min(start_index + max_span_width, len(sentence))
            for end_index in range(first_end, last_end):
                spans.append((start_index, end_index))
        return spans

    def __len__(self):
        return 100000000  # Dummy infinite-like length to allow random sampling

    def __getitem__(self, index):
        """
        Return a random training example (raw + tokenized).
        """
        index = random.randint(0, len(self.training_data) - 1)  # Sample randomly
        raw_data = self.training_data[index]
        data = raw_data.deepcopy_all_data()                     # Copy to avoid side effects

        tokens, idx_dict = self.encoder.tokenize(data['sentence'])  # Subword tokenize
        converted_spans = [
            self.convert_span(span, idx_dict)
            for span in data['spans']
        ]
        data['sentence'] = tokens
        data['spans'] = converted_spans
        return [raw_data, data]

    def get_unlabel_item(self, index):
        """
        Similar to __getitem__, but for unlabeled data.
        """
        raw_data = self.unlabel_data[index]
        data = raw_data.deepcopy_all_data()
        tokens, idx_dict = self.encoder.tokenize(data['sentence'])
        converted_spans = [
            self.convert_span(span, idx_dict)
            for span in data['spans']
        ]
        data['sentence'] = tokens
        data['spans'] = converted_spans
        return [raw_data, data]

    def convert_span(self, span, idx_dict):
        """
        Convert word-level span to subword-level indices using idx_dict.
        """
        start_idx, end_idx = span
        span_idx = idx_dict[start_idx] + idx_dict[end_idx]  # Combine token lists
        if len(span_idx) == 0:  # Handle tokens dropped by tokenizer
            return (0, 0)
        return (min(span_idx), max(span_idx))

    def collate_fn(self, data):
        """
        Custom collate to batch examples with padding.
        """
        raw_data_b, data_b = zip(*data)    # Separate raw and processed
        data_b = list_to_dict(data_b)      # Convert list of dicts → dict of lists

        # Pad sentences to same length
        max_len = max(len(tokens) for tokens in data_b['sentence'])
        for tokens in data_b['sentence']:
            while len(tokens) < max_len:
                tokens.append(0)
        data_b['sentence'] = torch.LongTensor(data_b['sentence'])

        # Create token mask (1 for real tokens)
        data_b['mask'] = data_b['sentence'].ne(0).float()

        # Pad spans, ner_labels, soft_labels, span masks, etc., to same span count
        max_span_count = max(len(spans) for spans in data_b['spans']) or 1
        # Spans
        for spans in data_b['spans']:
            while len(spans) < max_span_count:
                spans.append((0, 0))
        data_b['spans'] = torch.LongTensor(data_b['spans'])
        # NER labels
        for labels in data_b['ner_labels']:
            while len(labels) < max_span_count:
                labels.append(0)
        data_b['ner_labels'] = torch.LongTensor(data_b['ner_labels'])
        # Soft labels
        for soft in data_b['soft_labels']:
            while len(soft) < max_span_count:
                soft.append([0] * self.ner_label.get_num())
        data_b['soft_labels'] = torch.FloatTensor(data_b['soft_labels'])
        # Span mask
        for mask in data_b['span_mask']:
            while len(mask) < max_span_count:
                mask.append(0)
        data_b['span_mask'] = torch.FloatTensor(data_b['span_mask'])
        # Span mask for loss
        for mask in data_b['span_mask_for_loss']:
            while len(mask) < max_span_count:
                mask.append(0)
        data_b['span_mask_for_loss'] = torch.FloatTensor(data_b['span_mask_for_loss'])

        return raw_data_b, data_b

    def update_dataset(self, new_data):
        """
        Add pseudo-labeled data to training set (semi-supervised).
        """
        self.training_data = deepcopy(self.label_data) + new_data

class MyDataLoader:
    """Iterator over unlabeled data batches (custom for semi-supervised)."""

    def __init__(self, dataset, batch_size):
        self.dataset = dataset                   # Reference to DataSet
        self.batch_size = batch_size             # Batch size
        self.index = 0                           # Batch index
        self.max_batch_i = len(dataset.unlabel_data) // batch_size  # Full batches
        self.max_one_i = len(dataset.unlabel_data)                # Total examples

    def __iter__(self):
        self.index = 0
        return self

    def __len__(self):
        return self.max_one_i

    def __next__(self):
        """
        Return the next batch of unlabeled items or stop iteration.
        """
        if not self.dataset.unlabel_data:
            raise StopIteration

        batch = []
        if self.index < self.max_batch_i:
            start = self.index * self.batch_size
            end = (self.index + 1) * self.batch_size
        elif self.index == self.max_batch_i:
            start = self.index * self.batch_size
            end = self.max_one_i
        else:
            raise StopIteration

        for i in range(start, end):
            batch.append(self.dataset.get_unlabel_item(i))

        self.index += 1
        return self.dataset.collate_fn(batch)

    def reset(self):
        """Reset to start of data."""
        self.index = 0

    def has_next(self):
        """Check if more batches remain."""
        return self.index * self.batch_size < self.max_one_i

def get_loader(root, filename, encoder, batch_size, ner_label, is_train, opt, mode='training'):
    """
    Create PyTorch DataLoader for labeled data and MyDataLoader for unlabeled.
    Returns iterators depending on train/eval mode.
    """
    dataset = DataSet(root, filename, encoder, batch_size, ner_label, is_train=is_train, opt=opt, mode=mode)
    label_loader = data.DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=False,
        pin_memory=True,
        num_workers=8,
        collate_fn=dataset.collate_fn
    )
    unlabel_loader = MyDataLoader(dataset=dataset, batch_size=batch_size)
    if is_train:
        return iter(label_loader), iter(unlabel_loader)
    else:
        return iter(label_loader)

def update_train_loader(dataset, batch_size):
    """
    Re-create labeled and unlabeled loaders after expanding the training set.
    """
    label_loader = data.DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=False,
        pin_memory=True,
        num_workers=8,
        collate_fn=dataset.collate_fn
    )
    unlabel_loader = MyDataLoader(dataset=dataset, batch_size=batch_size)
    return iter(label_loader), iter(unlabel_loader)
