import yaml


def create_hparam_yml(TRAIN_BATCH_SIZE, TRAIN_EPOCHS, LEARNING_RATE, WEIGHT_DECAY, NUM_ATTENTION_HEADS, NUM_HIDDEN_LAYERS, save_to="hparams.yml"):
    # Hyperparameters
    hparams = {}
    set_no = 0
    for batch_size in TRAIN_BATCH_SIZE:
        for num_epoch in TRAIN_EPOCHS:
            for lr in LEARNING_RATE:
                for wd in WEIGHT_DECAY:
                    for num_heads in NUM_ATTENTION_HEADS:
                        for num_layers in NUM_HIDDEN_LAYERS:
                            hparams["set_" + str(set_no)] = {
                                "TRAIN_BATCH_SIZE": batch_size,
                                "VALID_BATCH_SIZE": 8,
                                "TRAIN_EPOCHS": num_epoch,
                                "LEARNING_RATE": lr,
                                "WEIGHT_DECAY": wd,
                                "MAX_LEN": 128,
                                "VOCAB_SIZE": 800,
                                "MAX_POSITION_EMBEDDINGS": 514,
                                "NUM_ATTENTION_HEADS": num_heads,
                                "NUM_HIDDEN_LAYERS": num_layers,
                                "TYPE_VOCAB_SIZE": 1,
                                "HIDDEN_SIZE": 768,
                            }
                            set_no += 1
                        set_no += 1
                    set_no += 1
                set_no += 1
            set_no += 1
        set_no += 1

    # Write to yaml file
    with open(save_to, "w") as f:
        yaml.dump(hparams, f)


create_hparam_yml(TRAIN_BATCH_SIZE=[16, 32, 64], TRAIN_EPOCHS=[5, 10], LEARNING_RATE=[1e-5], WEIGHT_DECAY=[0.001], NUM_ATTENTION_HEADS=[4, 8], NUM_HIDDEN_LAYERS=[8, 12])
