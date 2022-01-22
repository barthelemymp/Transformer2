import torch
import torch.nn as nn
import torch.optim as optim
#import spacy
import math
 
class PositionalEncoding(nn.Module):
    r"""Inject some information about the relative or absolute position of the tokens
        in the sequence. The positional encodings have the same dimension as
        the embeddings, so that the two can be summed. Here, we use sine and cosine
        functions of different frequencies.
    .. math::
        \text{PosEncoder}(pos, 2i) = sin(pos/10000^(2i/d_model))
        \text{PosEncoder}(pos, 2i+1) = cos(pos/10000^(2i/d_model))
        \text{where pos is the word position and i is the embed idx)
    Args:
        d_model: the embed dim (required).
        dropout: the dropout value (default=0.1).
        max_len: the max. length of the incoming sequence (default=5000).
    Examples:
        >>> pos_encoder = PositionalEncoding(d_model)
    source: https://github.com/pytorch/tutorials/blob/011ae8a6d47a960935d0401acda71d0e400088d6/advanced_source/ddp_pipeline.py#L43

    """

    def __init__(self, d_model, dropout=0.1, max_len=40):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        #pe = pe[:,:-1] 
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        r"""Inputs of forward function
        Args:
            x: the sequence fed to the positional encoder model (required).
        Shape:
            x: [sequence length, batch size, embed dim]
            output: [sequence length, batch size, embed dim]
        Examples:
            >>> output = pos_encoder(x)
        """
        #print( self.pe[:x.size(0), :].shape)
        #print(x.shape) #Embedding
        #print(self.pe[:, :].shape) #Positional Encoding
        x = x + self.pe[:x.shape[0], :]
        #print(x.shape)
        return self.dropout(x)


    
class Transformer(nn.Module):
    def __init__(
        self,
        embedding_size,
        src_vocab_size,
        trg_vocab_size,
        src_pad_idx,
        num_heads,
        num_encoder_layers,
        num_decoder_layers,
        forward_expansion,
        dropout,
        len_input,
	len_ouput,
        device,
    ):
        super(Transformer, self).__init__()
        self.src_word_embedding = nn.Embedding(src_vocab_size, embedding_size) #amino acid embedding
        self.trg_word_embedding = nn.Embedding(src_vocab_size, embedding_size)        
        
        self.src_position_embedding = PositionalEncoding(embedding_size,max_len=len_input) #position (lensent
        self.trg_position_embedding = PositionalEncoding(embedding_size,max_len=len_ouput)
        
        
        self.embedding_size = embedding_size
        self.device = device
        self.transformer = nn.Transformer(
            embedding_size,
            num_heads,
            num_encoder_layers,
            num_decoder_layers,
            forward_expansion,
            dropout,
        )
        self.fc_out = nn.Linear(embedding_size, trg_vocab_size)
        self.dropout = nn.Dropout(dropout)
        self.src_pad_idx = src_pad_idx
    
    
    def make_src_mask(self, src):
        """
        If we have padded the source input (to be of the same size among the same batch I guess)
        there is no need to do computation for them, so this function masks the 
        padded parts.
        src is sequence to the encoder 
        """
        src_mask = src.transpose(0, 1) == self.src_pad_idx

        # (N, src_len)
        return src_mask.to(self.device)

    def forward(self, src, trg):
        src_seq_length, N = src.shape
        trg_seq_length, N = trg.shape
        
      
        
        # Creating the input of the Encoder and Decoder
        # It considers also the position
        embed_src = self.src_position_embedding.forward(self.src_word_embedding(src))
        
        embed_trg = self.trg_position_embedding.forward(self.trg_word_embedding(trg))
        # Creating the mask for the source and for the target (note that
        # for the target we are using a built-in torch function)
        src_padding_mask = self.make_src_mask(src)
        trg_mask = self.transformer.generate_square_subsequent_mask(trg_seq_length).to(
            self.device
        )
	#pytorch module
        out = self.transformer(
            embed_src,
            embed_trg,
            src_key_padding_mask=src_padding_mask,
            tgt_mask=trg_mask,
        )
        out = self.fc_out(out)
        return out
