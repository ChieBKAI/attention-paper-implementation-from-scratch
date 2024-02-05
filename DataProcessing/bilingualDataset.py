import torch
from torch.utils.data import Dataset
from DataProcessing.causalMask import get_causal_mask


# Bilingual Dataset
class BilingualDataset(Dataset):

    def __init__(self, ds_raw, tokenizer_src, tokenizer_tgt, src_lang, tgt_lang, seq_len: int) -> None:
        self.ds_raw = ds_raw
        self.tokenizer_src = tokenizer_src
        self.tokenizer_tgt = tokenizer_tgt
        self.src_lang = src_lang
        self.tgt_lang = tgt_lang
        self.seq_len = seq_len
        
        self.sos_token = torch.tensor([tokenizer_tgt.token_to_id("[SOS]")], dtype=torch.int64)
        self.eos_token = torch.tensor([tokenizer_tgt.token_to_id("[EOS]")], dtype=torch.int64)
        self.pad_token = torch.tensor([tokenizer_tgt.token_to_id("[PAD]")], dtype=torch.int64)

    def __len__(self):
        return len(self.ds_raw)
    
    def __getitem__(self, idx):
        src = self.ds_raw[idx]['translation'][self.src_lang]
        tgt = self.ds_raw[idx]['translation'][self.tgt_lang]

        encode_tokens = self.tokenizer_src.encode(src).ids
        decode_tokens = self.tokenizer_tgt.encode(tgt).ids

        # Length of padding tokens
        encode_padding_len = self.seq_len - len(encode_tokens) - 2
        decode_padding_len = self.seq_len - len(decode_tokens) - 1

        # Raising error if the length of the tokens is greater than the seq_len
        if encode_padding_len < 0 or decode_padding_len < 0:
            raise Exception("Seq_len is too short for the tokens")

        encoder_input = torch.cat(
            [self.sos_token, 
            torch.tensor(encode_tokens, dtype=torch.int64),
            self.eos_token,
            torch.tensor([self.pad_token] * encode_padding_len, dtype=torch.int64)],
            dim=0
        )
        decoder_input = torch.cat(
            [self.sos_token, 
            torch.tensor(decode_tokens, dtype=torch.int64),
            torch.tensor([self.pad_token] * decode_padding_len, dtype=torch.int64)],
            dim=0
        )
        label = torch.cat(
            [torch.tensor(decode_tokens, dtype=torch.int64),
            self.eos_token,
            torch.tensor([self.pad_token] * decode_padding_len, dtype=torch.int64)],
            dim=0
        )

        # Check the size of the tensors to make sure they are all seq_len long
        assert encoder_input.size(0) == self.seq_len, f"encoder_input size is {encoder_input.size(0)} instead of {self.seq_len}"
        assert decoder_input.size(0) == self.seq_len, f"decoder_input size is {decoder_input.size(0)} instead of {self.seq_len}"
        assert label.size(0) == self.seq_len, f"label size is {label.size(0)} instead of {self.seq_len}"
        

        return {
            'encoder_input': encoder_input, # (seq_len)
            'decoder_input': decoder_input, # (seq_len)
            'encoder_mask': (encoder_input != self.pad_token).unsqueeze(0).unsqueeze(0).int(), # (1, 1, seq_len)
            'decoder_mask': (decoder_input != self.pad_token).unsqueeze(0).unsqueeze(0).int() & get_causal_mask(decoder_input.size(0)),
            'label': label, # (seq_len)
            'src': src,
            'tgt': tgt
        }