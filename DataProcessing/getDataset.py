import torch
from datasets import load_dataset
from DataProcessing.tokenizer import get_tokenizer
from DataProcessing.bilingualDataset import BilingualDataset
from config import get_config

# Get Dataset
def get_dataset(config):
    ds_raw = load_dataset('opus_books', f'{config["src_lang"]}-{config["tgt_lang"]}', split='train')

    # Max length of the sentences
    max_len = 0
    for item in ds_raw:
        max_len = max(max_len, len(item['translation'][config['src_lang']]), len(item['translation'][config['tgt_lang']]))

    print(f"Max length of the sentences: {max_len}")
    
    tokenizer_src = get_tokenizer(config, ds_raw, config['src_lang'])
    tokenizer_tgt = get_tokenizer(config, ds_raw, config['tgt_lang'])

    # Data split
    train_size = int(0.9 * len(ds_raw))
    test_size = len(ds_raw) - train_size
    ds_train, ds_test = torch.utils.data.random_split(ds_raw, [train_size, test_size])
    
    ds_train = BilingualDataset(ds_train, tokenizer_src, tokenizer_tgt, config['src_lang'], config['tgt_lang'], config['seq_len'])
    ds_test = BilingualDataset(ds_test, tokenizer_src, tokenizer_tgt, config['src_lang'], config['tgt_lang'], config['seq_len'])

    train_data_loader = torch.utils.data.DataLoader(ds_train, batch_size=config['batch_size'], shuffle=True)
    test_data_loader = torch.utils.data.DataLoader(ds_test, batch_size=config['batch_size'], shuffle=True)

    return train_data_loader, test_data_loader, tokenizer_src, tokenizer_tgt
