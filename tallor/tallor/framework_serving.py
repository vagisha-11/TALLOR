import os                                      # File‐path utilities
import sys                                     # System utilities (unused here)
from .data_loader import update_train_loader   # Rebuilds DataLoader after adding new data
import torch                                   # PyTorch for model, tensors
from transformers import AdamW, get_linear_schedule_with_warmup
                                              # Optimizer + learning‐rate schedule
from copy import deepcopy                      # Deep copy to avoid side effects
from collections import defaultdict            # Dict with automatic list creation
from tallor.rule_kits.rule_labeler import RuleLabeler
                                              # Rule‐based labeler for initial annotations
from tqdm import tqdm                          # Progress bars
import json                                    # JSON I/O


class IEFramework:
    """
    Semi-supervised IE framework:
    1. Apply dictionary/rule labeling (RuleLabeler).
    2. Train neural NER model on labeled set.
    3. Predict on unlabeled set.
    4. Select high‐confidence spans, add them to training.
    5. Repeat.
    """

    def __init__(self, ner_label, dataset_name, opt, logger,
                 batch_size, train_data_loader, unlabel_data_loader):
        self.train_data_loader = train_data_loader          # Iterator over labeled batches
        self.unlabel_data_loader = unlabel_data_loader      # Custom iterator over unlabeled
        self.training_set = self.unlabel_data_loader.dataset
        self.batch_size = batch_size
        # Initialize rule‐based labeler on all unlabeled data-labels unlabeled data
        self.Labeler = RuleLabeler(ner_label,
                                   self.training_set.unlabel_data,
                                   dataset_name, opt, mode='serving')
        self.ner_label = ner_label
        self.logger = logger

    def __load_model__(self, ckpt):
        """
        Load a PyTorch checkpoint.
        Raises if path not found.
        """
        if os.path.isfile(ckpt):
            checkpoint = torch.load(ckpt)
            print(f"Successfully loaded checkpoint '{ckpt}'")
            return checkpoint
        else:
            raise Exception(f"No checkpoint found at '{ckpt}'")

    def train(self, model, model_name,
              epoch=20, train_step=1500, load_ckpt=None,
              save_ckpt=None, warmup_step=100,
              update_threshold=0.7, result_dir=None):
        """
        Orchestrates training loop:
        - Optionally load pretrained weights.
        - Save initial model.
        - For each epoch:
            1. Rule‐label unlabeled data.
            2. Update dataset and DataLoader.
            3. Restore initial weights.
            4. Train NER model for train_step iterations.
            5. On last epoch: save final model, rules, and predictions.
            6. Otherwise: self‐training selection + metric reset.
        """

        # 1. Load previous checkpoint if provided
        if load_ckpt:
            state_dict = self.__load_model__(load_ckpt)['state_dict']
            own_state = model.state_dict()
            for name, param in state_dict.items():
                if name in own_state:
                    own_state[name].copy_(param)

        self.save_initial_model(model, save_ckpt)  # Backup initial weights
        best_ner_f1 = 0        
        self.logger.info('Start training!')
        rule_recoder = {}                           # Store extracted rules

        for i in range(epoch):
            print('Epoch:', i)
            # (a) Rule‐based pipeline returns new labeled examples
            rule_labeled_data, all_data = self.Labeler.pipeline(i, rule_recoder)
            # (b) Add rule‐labeled spans to training set + recreate DataLoaders
            self.update_dataset_and_loader(self.training_set, rule_labeled_data)
            # (c) Reset model to initial weights
            self.load_initial_model(model, save_ckpt)
            train_step += 50
            # (d) Train neural NER on combined labeled set
            train_f1, train_p, train_r = self.train_ner_model(
                                          model, train_step,
                                          warmup_step, best_ner_f1)

            if i == epoch - 1:  # Final epoch: save outputs
                torch.save({'state_dict': model.state_dict()}, save_ckpt)
                self.save_rules(result_dir, rule_recoder)
                self.predict_and_save_dataset(result_dir, model)
            else:
                # (e) Self‐train: select high-confidence predictions
                self.select_and_update_training(model, update_threshold)
                model.metric_reset()  # Reset evaluation metrics

        self.logger.info(f"Finish training {model_name}")

    def train_ner_model(self, model, train_iter, warmup_step, best_ner_f1):
        """
        One phase of NER model training:
        - Set up optimizer (AdamW) & scheduler.
        - For train_iter steps:
            • Fetch labeled batch.
            • Forward, compute loss, backward.
            • Clip gradients, step optimizer & scheduler.
            • Track metrics (precision, recall, F1).
        Returns final (F1, P, R).
        """
        model.train()
        # Split parameters into weight_decay vs no_decay groups
        named_params = [(n, p) for n, p in model.named_parameters() if p.requires_grad]
        no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
        optim_groups = [
            {'params': [p for n, p in named_params if not any(nd in n for nd in no_decay)],
             'weight_decay': 0.01},
            {'params': [p for n, p in named_params if  any(nd in n for nd in no_decay)],
             'weight_decay': 0.0}
        ]
        optimizer = AdamW(optim_groups, lr=2e-5, correct_bias=False)
        scheduler = get_linear_schedule_with_warmup(
                    optimizer, num_warmup_steps=warmup_step,
                    num_training_steps=train_iter)

        for _ in tqdm(range(train_iter), ncols=100, desc='Train NER model'):
            raw_data_b, data_b = next(self.train_data_loader)
            if torch.cuda.is_available():
                for k, v in data_b.items():
                    data_b[k] = v.cuda()

            # Forward pass: sentence tokens, masks, spans, labels, soft labels
            output_dict = model(data_b['sentence'], data_b['mask'],
                                data_b['spans'], data_b['span_mask'],
                                data_b['span_mask_for_loss'],
                                data_b['ner_labels'], data_b['soft_labels'])
            loss = output_dict['loss']
            loss.backward()                                 # Backpropagate loss
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5)  # Clip gradients
            optimizer.step()                                # Update parameters
            scheduler.step()                                # Update learning rate
            optimizer.zero_grad()

            # Optionally gather metrics (not used here)
            ner_results = output_dict['span_metrics']
        # Return precision, recall, F1 from last iteration
        prf = ner_results[1].get_metric()
        return prf['f'], prf['p'], prf['r']

    def select_and_update_training(self, model, update_threshold):
        """
        Perform self‐training:
        1. Predict on all unlabeled data.
        2. Select spans with confidence > threshold.
        3. Update rule pipeline with these new pseudo‐labels.
        """
        model.eval()
        new_data, raw_data = self.self_training(model, update_threshold)
        self.Labeler.update_rule_pipeline(new_data)

    def update_dataset_and_loader(self, dataset, new_data):
        """
        Add new_data to dataset.training_data and rebuild DataLoaders.
        """
        dataset.update_dataset(new_data)
        del self.train_data_loader
        # Recreate both labeled and unlabeled iterators
        self.train_data_loader, self.unlabel_data_loader = \
            update_train_loader(dataset, self.batch_size)
        self.logger.info(
            f'Update successfully! {len(dataset.training_data)} instances '
            f'for training, {len(new_data)} instances are new labeled.')

    def self_training(self, model, update_threshold):
        """
        Iterate over unlabeled batches, predict spans, decode them,
        then select and return those above confidence threshold.
        """
        raw_data, ner_res = [], []
        print('Begin predict all data.')
        while self.unlabel_data_loader.has_next():
            raw_b, data_b = next(self.unlabel_data_loader)
            if torch.cuda.is_available():
                for k, v in data_b.items():
                    data_b[k] = v.cuda()
            output = model.predict(data_b['sentence'], data_b['mask'],
                                   data_b['spans'], data_b['span_mask'])
            ner_res_list = model.decode(output)
            raw_data += raw_b
            ner_res += ner_res_list
        print('Done.')
        # Filter by threshold and return new pseudo‐labeled data
        new_data = self.select_and_update_data(raw_data, ner_res, update_threshold)
        return new_data, raw_data

    def select_and_update_data(self, raw_data, ner_res, update_threshold):
        """
        Group predictions by class, sort by probability, pick top fraction,
        then update each raw_data entry’s labels and span_mask_for_loss.
        """
        # 1. Bucket predictions per class
        class_buckets = defaultdict(list)
        for i, res_list in enumerate(ner_res):
            for ner in res_list:
                ner['data_id'] = i
                class_buckets[ner['class']].append(ner)

        # 2. For each class, sort by prob desc and take top fraction
        data_id_map = defaultdict(list)
        for cls, preds in class_buckets.items():
            preds.sort(key=lambda x: x['prob'], reverse=True)
            cutoff = int(len(preds) * update_threshold)
            for inst in preds[:cutoff]:
                data_id_map[inst['data_id']].append(inst)

        # 3. Apply selected pseudo‐labels to deep‐copied raw_data
        new_data = []
        for idx, entry in enumerate(deepcopy(raw_data)):
            updates = data_id_map.get(idx, [])
            new_data.append(self.update_data_entry(entry, updates))
        return new_data

    def update_data_entry(self, data_entry, ner_res_list):
        """
        Given a DataPoint and its new ner_res_list, set those spans’ labels
        and enable them in span_mask_for_loss (so they’re used in training).
        """
        for ner in ner_res_list:
            span_idx = ner['span_idx']
            label = ner['class']
            data_entry.ner_labels[span_idx] = label
            data_entry.span_mask_for_loss[span_idx] = 1
        return data_entry

    def save_initial_model(self, model, path):
        """Persist the model’s initial weights for later reset."""
        torch.save({'state_dict': model.state_dict()}, path + '_initial')

    def load_initial_model(self, model, path):
        """Restore model weights saved by save_initial_model."""
        sd = self.__load_model__(path + '_initial')['state_dict']
        own_state = model.state_dict()
        for name, param in sd.items():
            if name in own_state:
                own_state[name].copy_(param)

    def save_rules(self, dir, rule_recoder):
        """Dump extracted rules into JSON files."""
        path = os.path.join(dir, 'extracted_rules.json')
        composed, token_rules = defaultdict(list), defaultdict(list)
        for body, label in rule_recoder.items():
            if isinstance(body, tuple):
                composed[label].append(body)
            else:
                token_rules[label].append(body)
        json.dump({'composed_rules': composed,
                   'string_token_rules': token_rules},
                  open(path, 'w', encoding='utf8'))

    def predict_and_save_dataset(self, dir, model):
        """
        Final pass: predict on all unlabeled data, decode entities,
        and write {'sentence','tokens','entities'} per line.
        """
        raw_data, ner_res = [], []
        print('Begin predict and save all data.')
        while self.unlabel_data_loader.has_next():
            raw_b, data_b = next(self.unlabel_data_loader)
            if torch.cuda.is_available():
                for k, v in data_b.items():
                    data_b[k] = v.cuda()
            output = model.predict(data_b['sentence'], data_b['mask'],
                                   data_b['spans'], data_b['span_mask'])
            ner_list = model.decode(output)
            raw_data += raw_b
            ner_res += ner_list

        # Write predictions to ner_results.json
        path = os.path.join(dir, 'ner_results.json')
        with open(path, 'w', encoding='utf8') as f:
            for dp, ents in zip(raw_data, ner_res):
                sent = dp.sentence
                tokens = model.sentence_encoder.tokenize_to_string(sent)
                decoded = []
                for e in ents:
                    cat = self.ner_label.get_label(e['class'])
                    if cat:
                        decoded.append({'span': e['span'],
                                        'prob': e['prob'],
                                        'category': cat})
                f.write(json.dumps({'sentence': ' '.join(sent),
                                    'tokens': tokens,
                                    'entities': decoded}) + '\n')
        print('Done.')
