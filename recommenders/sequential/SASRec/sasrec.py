import torch
from recommenders.sequential.SASRec.transformer_decoder import TransformerBlock


class SASRec(torch.nn.Module):
    def __init__(self, config):
        super(SASRec, self).__init__()
        self.config = config
        self.num_items = self.config.num_items
        self.sequence_length = self.config.sequence_length
        self.embedding_dim = self.config.embedding_dim
        self.embeddings_dropout = torch.nn.Dropout(self.config.dropout_rate)

        self.num_heads = self.config.num_heads

        self.item_embedding = torch.nn.Embedding(self.num_items + 2, self.embedding_dim)
        self.position_embedding = torch.nn.Embedding(self.sequence_length, self.embedding_dim)

        self.transformer_blocks = torch.nn.ModuleList([
            TransformerBlock(self.embedding_dim, self.num_heads, self.embedding_dim, self.config.dropout_rate)
            for _ in range(self.config.num_blocks)
        ])
        self.seq_norm = torch.nn.LayerNorm(self.embedding_dim)
        self.output_embedding = torch.nn.Embedding(self.num_items + 2, self.embedding_dim)

    def get_output_embeddings(self) -> torch.nn.Embedding:
        return self.output_embedding

    #returns last hidden state and the attention weights
    def forward(self, input):
        seq = self.item_embedding(input.long())
        mask = (input != self.num_items).float().unsqueeze(-1)

        bs = seq.size(0)
        positions = torch.arange(seq.shape[1]).unsqueeze(0).repeat(bs, 1).to(input.device)
        pos_embeddings = self.position_embedding(positions)[:input.size(0)]
        seq = seq + pos_embeddings
        seq = self.embeddings_dropout(seq)
        seq *= mask

        attentions = []
        for block in self.transformer_blocks:
            seq, attention = block(seq, mask)
            attentions.append(attention)

        seq_emb = self.seq_norm(seq)
        return seq_emb, attentions

    def get_logits(self, model_out):
        output_embeddings = self.get_output_embeddings()
        scores = torch.einsum('bse, de -> bsd', model_out, output_embeddings.weight)
        scores[:, :, self.num_items:] = float("-inf")
        return scores.to(torch.float64)


class SASRecConfig(object):
    def __init__(self,
                 num_items,
                 sequence_length=200,
                 embedding_dim=64,
                 num_heads=2,
                 num_blocks=3,
                 dropout_rate=0.2
                 ):
        self.num_items = num_items
        self.sequence_length = sequence_length
        self.embedding_dim = embedding_dim
        self.num_heads = num_heads
        self.num_blocks = num_blocks
        self.dropout_rate = dropout_rate
