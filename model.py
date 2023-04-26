import torch
from torch.nn import TransformerEncoder, TransformerEncoderLayer
from helper import *

class Hyper_InteractE(torch.nn.Module):
    def __init__(self, params):
        super(Hyper_InteractE, self).__init__()

        self.p = params

        self.ent_embed = torch.nn.Embedding(self.p.num_ent, self.p.embed_dim)
        xavier_normal_(self.ent_embed.weight)
        self.rel_embed = torch.nn.Embedding(self.p.num_rel * 2 + 1, self.p.embed_dim)
        xavier_normal_(self.rel_embed.weight)

        self.bceloss = torch.nn.BCELoss()

        self.inp_drop = torch.nn.Dropout(self.p.inp_drop)
        self.hidden_drop = torch.nn.Dropout(self.p.hid_drop)
        self.feature_map_drop = torch.nn.Dropout2d(self.p.feat_drop)
        self.aware_drop = torch.nn.Dropout(0.1)

        self.chequer_perm = self.get_chequer_perm()

        self.w_sub = get_param((self.p.embed_dim * 2, 1))
        self.w_rel = get_param((self.p.embed_dim * 2, 1))

        self.leakyrelu = torch.nn.LeakyReLU(0.2)

        self.bn0 = torch.nn.BatchNorm2d(self.p.iperm)
        self.bn1 = torch.nn.BatchNorm2d(self.p.channel * self.p.iperm)
        self.bn2 = torch.nn.BatchNorm1d(self.p.embed_dim)

        self.register_parameter('bias', Parameter(torch.zeros(self.p.num_ent)))
        self.register_parameter('conv_filt',
                                Parameter(torch.zeros(self.p.channel, 1, self.p.filter_size, self.p.filter_size)))
        xavier_normal_(self.conv_filt) - 1

        self.flat_sz = self.p.ik_h * 2 * self.p.ik_w * self.p.channel * self.p.iperm
        self.fc = torch.nn.Linear(self.flat_sz + 198 * 32, self.p.embed_dim)

        if self.p.method == 'perceptual':
            self.mlp = torch.nn.Linear(self.p.embed_dim * 2, self.p.embed_dim)
        elif self.p.method == 'co-aware':
            self.w_method = get_param((self.p.embed_dim, 1))
            self.w_sub_aware = torch.nn.Linear(self.p.embed_dim, self.p.embed_dim)
            self.w_rel_aware = torch.nn.Linear(self.p.embed_dim, self.p.embed_dim)

        self.cov1 = torch.nn.Conv2d(1, 32, (3, 1))
        self.cov2 = torch.nn.Conv2d(32, 32, (1, 3))

        print('Hyper-InteractE')

    def loss(self, pred, true_label=None):
        loss = self.bceloss(pred, true_label)
        return loss

    def agg(self, sub_emb, rel_emb, quals_ent_emb, quals_rel_emb, N):

        nums = N.sum(dim=1)
        nums[nums == 0.0] = 1.0
        q_e = quals_ent_emb.sum(dim=1)
        q_r = quals_rel_emb.sum(dim=1)

        q_e = q_e.div(nums.unsqueeze(-1))
        q_r = q_r.div(nums.unsqueeze(-1))

        q_e = q_e * self.p.alpha + (1 - self.p.alpha) * sub_emb
        q_r = q_r * self.p.alpha + (1 - self.p.alpha) * rel_emb

        q = self.common(q_e, q_r, self.p.method)

        q = q.to(torch.float)
        q_r = q_r.to(torch.float)
        q_e = q_e.to(torch.float)
        matric = torch.cat([q_e.unsqueeze(1), q.unsqueeze(1), q_r.unsqueeze(1)], dim=1).unsqueeze(1)

        x = self.inp_drop(matric)
        x = self.cov1(x)
        x = self.feature_map_drop(x)
        x = self.cov2(x)
        x = self.feature_map_drop(x)
        x = x.view(-1, 198 * 32)

        return x

    def common(self, q_e, q_r, fusion='co-aware'):
        q_e = q_e.to(torch.float32)
        q_r = q_r.to(torch.float32)

        if fusion == 'mult':
            return q_e * q_r
        elif fusion == 'perceptual':
            score_emb = torch.cat([q_e, q_r], dim=-1)
            return self.mlp(score_emb)
        elif fusion == 'co-aware':
            sub_emb_score = torch.matmul(q_e, self.w_method).squeeze(-1)
            rel_emb_score = torch.matmul(q_r, self.w_method).squeeze(-1)
            sub_emb_score = -self.leakyrelu(sub_emb_score)
            rel_emb_score = -self.leakyrelu(rel_emb_score)

            sub_emb_score = torch.exp(sub_emb_score)
            rel_emb_score = torch.exp(rel_emb_score)
            score_all = sub_emb_score + rel_emb_score
            sub_emb_score = (sub_emb_score / score_all).unsqueeze(-1).repeat(1, self.p.embed_dim)
            rel_emb_score = (rel_emb_score / score_all).unsqueeze(-1).repeat(1, self.p.embed_dim)
            return sub_emb_score * q_e + rel_emb_score * q_r

    def circular_padding_chw(self, batch, padding):
        upper_pad = batch[..., -padding:, :]
        lower_pad = batch[..., :padding, :]
        temp = torch.cat([upper_pad, batch, lower_pad], dim=2)

        left_pad = temp[..., -padding:]
        right_pad = temp[..., :padding]
        padded = torch.cat([left_pad, temp, right_pad], dim=3)
        return padded

    def get_chequer_perm(self):
        ent_perm = np.int32([np.random.permutation(self.p.embed_dim) for _ in range(self.p.iperm)])
        rel_perm = np.int32([np.random.permutation(self.p.embed_dim) for _ in range(self.p.iperm)])

        comb_idx = []
        for k in range(self.p.iperm):
            temp = []
            ent_idx, rel_idx = 0, 0

            for i in range(self.p.ik_h):
                for j in range(self.p.ik_w):
                    if k % 2 == 0:
                        if i % 2 == 0:
                            temp.append(ent_perm[k, ent_idx])
                            ent_idx += 1
                            temp.append(rel_perm[k, rel_idx] + self.p.embed_dim)
                            rel_idx += 1
                        else:
                            temp.append(rel_perm[k, rel_idx] + self.p.embed_dim)
                            rel_idx += 1
                            temp.append(ent_perm[k, ent_idx])
                            ent_idx += 1
                    else:
                        if i % 2 == 0:
                            temp.append(rel_perm[k, rel_idx] + self.p.embed_dim)
                            rel_idx += 1
                            temp.append(ent_perm[k, ent_idx])
                            ent_idx += 1
                        else:
                            temp.append(ent_perm[k, ent_idx])
                            ent_idx += 1
                            temp.append(rel_perm[k, rel_idx] + self.p.embed_dim)
                            rel_idx += 1

            comb_idx.append(temp)

        chequer_perm = torch.LongTensor(np.int32(comb_idx))
        return chequer_perm

    def forward(self, sub, rel, quals, N):

        sub_emb = self.ent_embed(sub)
        rel_emb = self.rel_embed(rel)  # batch_size,100

        quals_ent_emb = self.ent_embed(quals[:, 1::2])
        quals_rel_emb = self.rel_embed(quals[:, 0::2])

        x2 = self.agg(sub_emb, rel_emb, quals_ent_emb, quals_rel_emb, N)

        comb_emb = torch.cat([sub_emb, rel_emb], dim=1)
        chequer_perm = comb_emb[:, self.chequer_perm]
        stack_inp = chequer_perm.reshape((-1, self.p.iperm, 2 * self.p.ik_w, self.p.ik_h))
        stack_inp = self.bn0(stack_inp)
        x = self.inp_drop(stack_inp)
        x = self.circular_padding_chw(x, self.p.filter_size // 2)
        x = F.conv2d(x, self.conv_filt.repeat(self.p.iperm, 1, 1, 1), padding=0, groups=self.p.iperm)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.feature_map_drop(x)
        x = x.view(-1, self.flat_sz)
        x = torch.cat([x, x2], dim=-1)
        x = self.fc(x)
        x = self.hidden_drop(x)
        x = self.bn2(x)
        x = F.relu(x)

        x = torch.mm(x, self.ent_embed.weight.transpose(1, 0))
        x += self.bias.expand_as(x)

        pred = torch.sigmoid(x)

        return pred
