import torch
from torch.utils.data.dataset import Dataset

class CustomDataset(Dataset):
    def __init__(self, df, tokenizer, MAX_LEN):
        self.examples = []
        
        for example in df.values:
            x = tokenizer.encode_plus(example, max_length = MAX_LEN, truncation=True, padding='max_length')
            self.examples += [x.input_ids]

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, i):
        # We’ll pad at the batch level.
        return torch.tensor(self.examples[i])


from transformers import RobertaConfig
from transformers import RobertaForMaskedLM

import pandas as pd

from transformers import RobertaTokenizerFast

from transformers import DataCollatorForLanguageModeling

from transformers import Trainer, TrainingArguments



def train_and_save_roberta_model(selfies_path="./data/selfies_subset.txt", save_folder="./saved_model/"):
    TRAIN_BATCH_SIZE = 16    # input batch size for training (default: 64)
    VALID_BATCH_SIZE = 8    # input batch size for testing (default: 1000)
    TRAIN_EPOCHS = 3        # number of epochs to train (default: 10)
    LEARNING_RATE = 1e-4    # learning rate (default: 0.001)
    WEIGHT_DECAY = 0.01
    SEED = 42               # random seed (default: 42)
    MAX_LEN = 128
    SUMMARY_LEN = 7

    config = RobertaConfig(
        vocab_size=8192,
        max_position_embeddings=514,
        num_attention_heads=2,
        num_hidden_layers=1,
        type_vocab_size=1,
    )

    model = RobertaForMaskedLM(config=config)
    df = pd.read_csv(selfies_path, header=None)

    tokenizer = RobertaTokenizerFast.from_pretrained("./data/bpe/")

    train_dataset = CustomDataset(df[0][:100], tokenizer, MAX_LEN) # column name is 0 temp.
    eval_dataset = CustomDataset(df[0][:100], tokenizer, MAX_LEN)

    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer, mlm=True, mlm_probability=0.15
    )

    training_args = TrainingArguments(
        output_dir=save_folder,
        overwrite_output_dir=True,
        evaluation_strategy = 'epoch',
        num_train_epochs=TRAIN_EPOCHS,
        learning_rate=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
        per_device_train_batch_size=TRAIN_BATCH_SIZE,
        per_device_eval_batch_size=VALID_BATCH_SIZE,
        save_steps=8192,
        #eval_steps=4096,
        save_total_limit=1,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        #prediction_loss_only=True,
    )

    trainer.train()
    trainer.save_model(save_folder)