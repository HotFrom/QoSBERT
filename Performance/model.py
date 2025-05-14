from typing import List
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.normal import Normal

class MCDropout(nn.Dropout):
    def forward(self, x):
        return F.dropout(x, self.p, training=True, inplace=False)

class AttnPooling(nn.Module):
    def __init__(self, hidden: int):
        super().__init__()
        self.score = nn.Linear(hidden, 1, bias=False)

    def forward(self, feats, mask):
        w = self.score(feats).squeeze(-1)
        w = w.masked_fill(~mask, -1e4)
        w = torch.softmax(w, dim=1)
        return torch.bmm(w.unsqueeze(1), feats).squeeze(1)

class GaussianHead(nn.Module):
    def __init__(self, in_dim: int):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, 512), nn.GELU(),
            nn.Linear(512, 128),    nn.GELU(),
            nn.Linear(128, 2)  # output: [μ, logσ²]
        )

    def forward(self, x):
        out = self.mlp(x)
        mu, logvar = out.split(1, dim=-1)
        var = F.softplus(logvar.clamp(min=-5.0, max=5.0)) + 1e-3
        return mu.squeeze(-1), var.squeeze(-1)

class Model(nn.Module):
    def __init__(
        self,
        encoder,
        config,
        tokenizer,
        args,
        num_last_layers: int = 4,
        dropout_p: float = 0.0,
        gradual_unfreeze: bool = False
    ):
        super().__init__()
        self.encoder      = encoder
        self.num_last     = num_last_layers
        self.attn_pool    = AttnPooling(config.hidden_size)
        self.mc_dropout   = MCDropout(dropout_p)
        self.pool_proj    = nn.Linear(3 * config.hidden_size, config.hidden_size)
        self.head         = GaussianHead(config.hidden_size)
        self.args         = args
        self.pad_id       = tokenizer.pad_token_id or tokenizer.eos_token_id
        self.log_temp     = nn.Parameter(torch.tensor(0.4055))  # log(1.5)

        self.total_layers: List[nn.Module] = self._find_transformer_layers()
        if gradual_unfreeze:
            self._freeze_all_encoder()
            for n, p in self.encoder.named_parameters():
                if any(k in n for k in ["LayerNorm", "layernorm", "ln_", "norm", "embed"]):
                    p.requires_grad = True

    @staticmethod
    def gaussian_nll(mu, var, y, eps: float = 1e-6):
        var = var + eps
        dist = Normal(mu, var.sqrt())
        return -(dist.log_prob(y)).mean()

    def apply_temperature(self, mu, var):
        tau2 = torch.exp(self.log_temp * 2)
        return mu, var * tau2.clamp(min=1e-6)

    def enable_mc_dropout(self):
        for m in self.encoder.modules():
            if isinstance(m, nn.Dropout):
                m.train()

    def forward(self, input_ids, labels=None, mc_times: int = 1, calib=False):
        if mc_times > 1:
            self.enable_mc_dropout()

        mask = input_ids.ne(self.pad_id)
        outs = self.encoder(input_ids,
                            attention_mask=mask,
                            output_hidden_states=True)

        hidden_stack = torch.stack(outs.hidden_states[-self.num_last:], dim=0) \
                            .permute(1, 2, 0, 3)  # [L, B, T, H] → [B, T, L, H]
        hidden_mean = hidden_stack.mean(dim=2)  # [B, T, H]

        mean_pool = hidden_mean.mean(1)
        max_pool  = hidden_mean.masked_fill(~mask.unsqueeze(-1), -1e4).max(1).values
        max_pool  = torch.clamp(max_pool, min=0)
        attn_pool = self.attn_pool(hidden_mean, mask)

        pooled = torch.cat([mean_pool, max_pool, attn_pool], dim=-1)
        seq_vec = self.pool_proj(pooled)
        seq_vec = self.mc_dropout(seq_vec)

        mu_s, var_s = [], []
        for _ in range(mc_times):
            mu_i, var_i = self.head(seq_vec)
            mu_s.append(mu_i)
            var_s.append(var_i)

        mu_stack = torch.stack(mu_s, dim=0)   # [T, B]
        var_stack = torch.stack(var_s, dim=0) # [T, B]

        mu = mu_stack.mean(0)
        aleatoric = var_stack.mean(0)
        epistemic = mu_stack.pow(2).mean(0) - mu.pow(2)
        var = aleatoric + epistemic

        if calib:
            mu, var = self.apply_temperature(mu, var)

        if labels is None:
            return mu, var

        nll_loss = self.gaussian_nll(mu, var, labels.float())
        mae_loss = F.l1_loss(mu, labels.float())
        total_loss =  mae_loss

        return total_loss, mu, var

    def _freeze_all_encoder(self):
        for p in self.encoder.parameters():
            p.requires_grad = False

    def unfreeze_encoder_gradually(self, n_layers: int = 1):
        if not self.total_layers:
            return
        for layer in self.total_layers[-n_layers:]:
            for p in layer.parameters():
                p.requires_grad = True
        self.total_layers = self.total_layers[:-n_layers]

    def _find_transformer_layers(self) -> List[nn.Module]:
        candidate_paths = [
            "encoder.layer",
            "transformer.h",
            "transformer.blocks",
            "h",
            "layers"
        ]
        for path in candidate_paths:
            obj = self.encoder
            ok = True
            for attr in path.split("."):
                if hasattr(obj, attr):
                    obj = getattr(obj, attr)
                else:
                    ok = False
                    break
            if ok and isinstance(obj, (list, nn.ModuleList)):
                return list(obj)
        return []
