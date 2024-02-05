import torch
import torch.nn as nn
from pathlib import Path
from model import build_model
from DataProcessing.getDataset import get_dataset
from torch.utils.tensorboard import SummaryWriter
from validation import run_validation
from tqdm import tqdm
from DataProcessing.filePath import get_weights_file_path



# Get Model
def get_model(config, tokenizer_src, tokenizer_tgt):
    transformer = build_model(
        src_vocab_size=tokenizer_src.get_vocab_size(),
        tgt_vocab_size=tokenizer_tgt.get_vocab_size(),
        src_seq_len=config['seq_len'],
        tgt_seq_len=config['seq_len'],
        d_model=config['d_model'],
        h=config['h'],
        d_ff=config['d_ff'],
        num_layers=config['num_layers'],
        dropout=config['dropout'],
    )
    return transformer

# Train model
def train_model(config):
    # Define device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Using device:', device)

    Path(config['model_folder']).mkdir(parents=True, exist_ok=True)

    # Get dataset
    train_data_loader, test_data_loader, tokenizer_src, tokenizer_tgt = get_dataset(config)

    # Get model
    transformer_model = get_model(config, tokenizer_src, tokenizer_tgt)
    transformer_model.to(device) # move model to device

    # Tensorboard
    writer = SummaryWriter(config['experiment_name'])

    # Optimizer
    optimizer = torch.optim.Adam(transformer_model.parameters(), lr=config['lr'], eps=1e-9)

    initial_epoch = 0
    global_step = 0
    if config['preload'] is not None:
        model_filename = get_weights_file_path(config, config['preload'])
        print('Loading model weights from:', model_filename)
        state = torch.load(model_filename)
        initial_epoch = state['epoch']
        optimizer.load_state_dict(state['optimizer_state_dict'])
        transformer_model.load_state_dict(state['model_state_dict'])
        global_step = state['global_step']

    # Loss function
    loss_fn = nn.CrossEntropyLoss(ignore_index=tokenizer_src.token_to_id('[PAD]'), label_smoothing=0.1).to(device)

    # Training loop
    for epoch in range(initial_epoch, config['epochs']):
        print('Epoch:', epoch)
        transformer_model.train()
        batch_iterator = tqdm(train_data_loader, desc='Processing epoch')
        for batch in batch_iterator:
            encoder_input = batch['encoder_input'].to(device)
            decoder_input = batch['decoder_input'].to(device)
            encoder_mask = batch['encoder_mask'].to(device)
            decoder_mask = batch['decoder_mask'].to(device)

            # Forward pass
            encoder_output = transformer_model.encode(encoder_input, encoder_mask)
            decoder_output = transformer_model.decode(decoder_input, encoder_output, encoder_mask, decoder_mask)
            output = transformer_model.project(decoder_output)

            # Compute loss
            label = batch['label'].to(device)
            loss = loss_fn(output.view(-1, tokenizer_tgt.get_vocab_size()), label.view(-1))
            batch_iterator.set_postfix({"loss": f"{loss.item():6.3f}"})

            # Log loss
            writer.add_scalar('train loss', loss.item(), global_step)
            writer.flush()

            # Backward pass
            loss.backward()

            # Update weights
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)

            global_step += 1

        # Run validation at the end of every epoch
        run_validation(transformer_model, test_data_loader, tokenizer_src, tokenizer_tgt, config['seq_len'], device, lambda msg: batch_iterator.write(msg), global_step, writer)

        # Save model
        model_filename = get_weights_file_path(config, config['epochs'])
        print('Saving model weights to:', model_filename)
        torch.save({
            'epoch': epoch + 1,
            'global_step': global_step,
            'model_state_dict': transformer_model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': loss
        }, model_filename)
