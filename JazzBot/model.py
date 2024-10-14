import torch.nn as nn
import math
from JazzBot.positional_encoding import *
from JazzBot.config import *
from JazzBot.vocab import itos_vocab
import pytorch_lightning as pl


class JazzBot(pl.LightningModule):
    def __init__(self, num_tokens, dim_model, num_heads, num_encoder_layers, num_decoder_layers, dropout_p, lr):
        super().__init__()

        # INFO
        self.model_type = "Transformer"
        self.dim_model = dim_model
        self.lr = lr
        self.criterion = nn.CrossEntropyLoss()

        # LAYERS
        self.positional_encoder = PositionalEncoding(dim_model=dim_model, dropout_p=dropout_p, max_len=5000)
        self.embedding = nn.Embedding(num_tokens, dim_model)
        self.transformer = nn.Transformer(
            d_model=dim_model,
            nhead=num_heads,
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers,
            dropout=dropout_p,
            batch_first = True
        )
        # One linear layer for each type of token (n, d, t, v)
        self.out1 = nn.Linear(dim_model, num_tokens)
        self.out2 = nn.Linear(dim_model, num_tokens)
        self.out3 = nn.Linear(dim_model, num_tokens)
        self.out4 = nn.Linear(dim_model, num_tokens)


    # A modifier pour utiliser 4 out functions différentes selon les cas    
    def forward(self, src, tgt, tgt_mask=None, src_pad_mask=None, tgt_pad_mask=None):
        # Src size must be (batch_size, src sequence length)
        # Tgt size must be (batch_size, tgt sequence length)

        # Embedding + positional encoding - Out size = (batch_size, sequence length, dim_model)
        prev_token = tgt[:,-1]
        src = self.embedding(src) * math.sqrt(self.dim_model)
        tgt = self.embedding(tgt) * math.sqrt(self.dim_model)
        src = self.positional_encoder(src)
        tgt = self.positional_encoder(tgt)
        
        transformer_out = self.transformer(src, tgt, tgt_mask=tgt_mask, src_key_padding_mask=src_pad_mask, tgt_key_padding_mask=tgt_pad_mask)

        
        type_tok = itos_vocab[prev_token[-1]][0]
        if type_tok =='n':
            out = self.out1(transformer_out)
        elif type_tok=='d':
            out = self.out2(transformer_out)
        elif type_tok=='t':
            out = self.out3(transformer_out)
        elif type_tok=='v':
            out = self.out4(transformer_out)
        
        return out

    def get_tgt_mask(self, size) -> torch.tensor:
        # Generates a squeare matrix where the each row allows one word more to be seen
        mask = torch.tril(torch.ones(size, size) == 1) # Lower triangular matrix
        mask = mask.float()
        mask = mask.masked_fill(mask == 0, float('-inf')) # Convert zeros to -inf
        mask = mask.masked_fill(mask == 1, float(0.0)) # Convert ones to 0
        
        return mask
    

    def configure_optimizers(self):
        optimizer = torch.optim.SGD(self.parameters(), lr=self.lr)
        return optimizer
    

    def training_step(self, batch):
        y_input, y_expected = batch
        X = torch.tensor(([0]*len(y_input)), device=self.device)
        y_input, y_expected = y_input.to(self.device), y_expected.to(self.device)

        # Get mask to mask out the next words
        sequence_length = y_input.size(1)
        tgt_mask = self.get_tgt_mask(sequence_length)

        # Standard training except we pass in y_input and tgt_mask
        pred = self(X, y_input, tgt_mask)

        # Permute pred to have batch size first again
        pred = pred.permute(0, 2, 1)
        loss = self.criterion(pred, y_expected)
        self.log("ptl/train_loss", loss)

        return loss
    
    
    def validation_step(self, batch, batch_idx):
        y_input, y_expected = batch
        X = torch.tensor(([0]*len(y_input)), device=self.device)
        y_input, y_expected = y_input.to(self.device), y_expected.to(self.device)

        # Get mask to mask out the next words
        sequence_length = y_input.size(1)
        tgt_mask = self.get_tgt_mask(sequence_length)

        # Standard training except we pass in y_input and tgt_mask
        pred = self(X, y_input, tgt_mask)

        # Permute pred to have batch size first again
        pred = pred.permute(0, 2, 1)
        loss = self.criterion(pred, y_expected)
        return {"val_loss": loss}

    def validation_epoch_end(self, outputs):
        avg_loss = torch.stack([x["val_loss"] for x in outputs]).mean()
        self.log("ptl/val_loss", avg_loss)