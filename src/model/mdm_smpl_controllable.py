import torch
import torch.nn as nn
from einops import repeat

from src.model.gaussian import GaussianDiffusion
from src.model.mdm_smpl import TransformerDenoiser




class TransformerDenoiserControllable(TransformerDenoiser):

    def __init__(
        self,nfeats,tx_dim,latent_dim,ff_size,num_layers,num_heads,dropout,nb_registers,activation
    ):
        super().__init__(nfeats,tx_dim,latent_dim,ff_size,num_layers,num_heads,dropout,nb_registers,activation)
        self.position_mlp = nn.Sequential(
            nn.Linear(2, 2 * latent_dim),
            nn.GELU(),
            nn.Linear(2 * latent_dim, latent_dim),
            nn.LayerNorm(latent_dim),
        )

    def forward(self, x, y, p, t):
        device = x.device
        x_mask = y["mask"]
        bs, nframes, nfeats = x.shape

        # Time embedding
        time_emb = self.timestep_encoder(t)
        time_mask = torch.ones(bs, dtype=bool, device=device)

        # put all the additionnal here
        info_emb = time_emb[:, None]
        info_mask = time_mask[:, None]

        assert "tx" in y

        # Condition part (can be text/action etc)
        tx_x = y["tx"]["x"]
        tx_mask = y["tx"]["mask"]

        tx_emb = self.tx_embedding(tx_x)

        pos_emb = self.position_mlp(p)
        pos_mask = torch.ones(bs, dtype=bool, device=device)[:, None]

        info_emb = torch.cat((info_emb, tx_emb), 1)
        info_mask = torch.cat((info_mask, tx_mask), 1)

        # add registers
        if self.nb_registers > 0:
            registers = repeat(self.registers, "nbtoken dim -> bs nbtoken dim", bs=bs)
            registers_mask = torch.ones((bs, self.nb_registers), dtype=bool, device=device)
            # add the register
            info_emb = torch.cat((info_emb, registers), 1)
            info_mask = torch.cat((info_mask, registers_mask), 1)

        x = self.skel_embedding(x)
        number_of_info = info_emb.shape[1] + 1

        # adding the embedding token for all sequences
        xseq = torch.cat((info_emb, x), 1)

        # add positional encoding to all the tokens
        xseq = self.sequence_pos_encoding(xseq)

        xseq = torch.cat((pos_emb.unsqueeze(1), xseq), 1)

        # create a bigger mask, to allow attend to time and condition as well
        aug_mask = torch.cat((info_mask, x_mask), 1)

        aug_mask = torch.cat((pos_mask, aug_mask), 1)

        final = self.seqTransEncoder(xseq, src_key_padding_mask=~aug_mask)

        # extract the important part
        output = self.to_skel_layer(final[:, number_of_info:])
        return output