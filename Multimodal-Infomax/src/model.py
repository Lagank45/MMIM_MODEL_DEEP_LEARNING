import torch
from torch import nn
import torch.nn.functional as F

from modules.encoders import LanguageEmbeddingLayer, CPC, MMILB, RNNEncoder, SubNet


# ---------------- GCN ---------------- #
class GCNFusion(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.linear = nn.Linear(dim, dim)

    def forward(self, t, a, v):
        H = torch.stack([t, a, v], dim=1)

        A = torch.matmul(H, H.transpose(1, 2)) / (H.size(-1) ** 0.5)
        A = torch.softmax(A, dim=-1)

        H_new = torch.matmul(A, H)

        # residual (important)
        H_new = H_new + H

        H_new = F.relu(self.linear(H_new))

        return torch.mean(H_new, dim=1)


# ---------------- TOKEN POOLING ---------------- #
class TokenPooling(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.fc1 = nn.Linear(dim, dim)
        self.fc2 = nn.Linear(dim, 1)

    def forward(self, x):
        h = torch.tanh(self.fc1(x))
        weights = torch.softmax(self.fc2(h), dim=1)
        return (weights * x).sum(dim=1)


# ---------------- MODEL ---------------- #
class MMIM(nn.Module):
    def __init__(self, hp):
        super().__init__()
        self.hp = hp
        self.add_va = hp.add_va
        hp.d_tout = hp.d_tin

        # Encoders
        self.text_enc = LanguageEmbeddingLayer(hp)

        self.visual_enc = RNNEncoder(
            hp.d_vin, hp.d_vh, hp.d_vout,
            hp.n_layer,
            hp.dropout_v if hp.n_layer > 1 else 0.0,
            hp.bidirectional
        )

        self.acoustic_enc = RNNEncoder(
            hp.d_ain, hp.d_ah, hp.d_aout,
            hp.n_layer,
            hp.dropout_a if hp.n_layer > 1 else 0.0,
            hp.bidirectional
        )

        # Token pooling
        self.text_pool = TokenPooling(hp.d_tout)

        # ---------------- MI ---------------- #
        self.mi_tv = MMILB(hp.d_tout, hp.d_vout,
                          hp.mmilb_mid_activation,
                          hp.mmilb_last_activation)

        self.mi_ta = MMILB(hp.d_tout, hp.d_aout,
                          hp.mmilb_mid_activation,
                          hp.mmilb_last_activation)

        # ---------------- COMMON DIM ---------------- #
        self.common_dim = hp.d_prjh

        self.proj_t = nn.Linear(hp.d_tout, self.common_dim)
        self.proj_a = nn.Linear(hp.d_aout, self.common_dim)
        self.proj_v = nn.Linear(hp.d_vout, self.common_dim)

        # ---------------- GCN ---------------- #
        self.gcn = GCNFusion(self.common_dim)

        # ---------------- GATING ---------------- #
        self.gate_t = nn.Linear(self.common_dim, self.common_dim)
        self.gate_a = nn.Linear(self.common_dim, self.common_dim)
        self.gate_v = nn.Linear(self.common_dim, self.common_dim)
        self.gate_g = nn.Linear(self.common_dim, self.common_dim)

        # reduce raw concat
        self.reduce_raw = nn.Linear(self.common_dim * 4, self.common_dim)

        # ---------------- NORMALIZATION ---------------- #
        self.norm = nn.LayerNorm(self.common_dim)
        self.dropout = nn.Dropout(0.2)

        # ---------------- CPC ---------------- #
        self.cpc_zt = CPC(hp.d_tout, hp.d_prjh, hp.cpc_layers, hp.cpc_activation)
        self.cpc_zv = CPC(hp.d_vout, hp.d_prjh, hp.cpc_layers, hp.cpc_activation)
        self.cpc_za = CPC(hp.d_aout, hp.d_prjh, hp.cpc_layers, hp.cpc_activation)

        # Classifier
        self.classifier = SubNet(
            in_size=self.common_dim,
            hidden_size=hp.d_prjh,
            n_class=hp.n_class,
            dropout=hp.dropout_prj
        )

    def forward(self, sentences, visual, acoustic, v_len, a_len,
                bert_sent, bert_sent_type, bert_sent_mask,
                y=None, mem=None):

        # -------- TEXT -------- #
        enc_word = self.text_enc(sentences, bert_sent, bert_sent_type, bert_sent_mask)
        text = self.text_pool(enc_word)

        # -------- AUDIO & VIDEO -------- #
        acoustic = self.acoustic_enc(acoustic, a_len)
        visual = self.visual_enc(visual, v_len)

        # -------- MI -------- #
        if y is not None:
            lld_tv, tv_pn, H_tv = self.mi_tv(text, visual, labels=y, mem=mem['tv'])
            lld_ta, ta_pn, H_ta = self.mi_ta(text, acoustic, labels=y, mem=mem['ta'])
        else:
            lld_tv, tv_pn, H_tv = self.mi_tv(text, visual)
            lld_ta, ta_pn, H_ta = self.mi_ta(text, acoustic)

        # -------- PROJECTION -------- #
        t = self.proj_t(text)
        a = self.proj_a(acoustic)
        v = self.proj_v(visual)

        # -------- MODALITY DROPOUT -------- #
        if self.training:
            mask = torch.rand(t.size(0), 4, device=t.device)
            mask = (mask > 0.1).float()

            t = t * mask[:, 0:1]
            a = a * mask[:, 1:2]
            v = v * mask[:, 2:3]

        # -------- GCN -------- #
        g = self.gcn(t, a, v)

        # -------- GATING -------- #
        gt = torch.sigmoid(self.gate_t(t))
        ga = torch.sigmoid(self.gate_a(a))
        gv = torch.sigmoid(self.gate_v(v))
        gg = torch.sigmoid(self.gate_g(g))

        fusion_gate = gt * t + ga * a + gv * v + gg * g

        # -------- HYBRID FUSION -------- #
        fusion_raw = torch.cat([t, a, v, g], dim=1)
        fusion_raw = self.reduce_raw(fusion_raw)

        # balanced fusion (no aggressive weighting)
        fusion = fusion_gate + fusion_raw

        # -------- NORMALIZE -------- #
        fusion = self.norm(fusion)
        fusion = self.dropout(fusion)

        # -------- PRED -------- #
        _, preds = self.classifier(fusion)

        # -------- CPC -------- #
        nce = (
            self.cpc_zt(text, fusion) +
            self.cpc_zv(visual, fusion) +
            self.cpc_za(acoustic, fusion)
        )

        pn_dic = {'tv': tv_pn, 'ta': ta_pn}
        lld = lld_tv + lld_ta
        H = H_tv + H_ta

        return lld, nce, preds, pn_dic, H