#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2020 Apple Inc. All Rights Reserved.
#

import torch
import torch.nn as nn

eps = 1e-20

class ProbabilisticAttention(nn.Module):
    def __init__(self, uniform_query_precision=False, uniform_value_precision=False, magnitude_priors=False,
                 key_adaptation=False, key_adaptation_iters=0, value_belief_propagation_iters=0):
        super(ProbabilisticAttention, self).__init__()
        self.uniform_query_precision = uniform_query_precision
        self.uniform_value_precision = uniform_value_precision
        self.magnitude_priors = magnitude_priors
        self.key_adaptation = key_adaptation
        self.key_adaptation_iters = key_adaptation_iters
        self.value_belief_propagation_iters = value_belief_propagation_iters

    def forward(self, q, zeta, alpha, mu, beta, pi=None, v_init=None, v_fixed=None,
                zeta_prior_precision=None, mu_prior_precision=None,
                q_pos_emb=None, zeta_pos_emb=None, v_pos_emb=None, nonzero_wts_mask=None):
        """
        Runs an update of the probabilistic version of attention based on a Mixture of Gaussians model.
        This layer is equivalent to a standard dot product attention when:
        self.uniform_query_precision = True
        self.uniform_value_precision = True
        sef.magnitude_priors = True
        alpha = 1/sqrt(C) (Could be a scalar to save some memory)
        beta = 0 (Could be a scalar to save some memory)
        v_init = None
        v_fixed = None
        :param q: A tensor of queries with dims N, G, C, H
        :param zeta: A tensor of keys (query/key Gaussian means) with dims N, G, C, H
        :param alpha: A scalar (see special case above) or tensor of query/key Gaussian precisions with dims N, G, C, H
        :param mu: A tensor of value Gaussian means with dims N, G, Cv, H
        :param beta: A scalar (see special case above) or tensor of value Gaussian precisions with dims N, G, C, H
        :param pi: A tensor of mixture component priors with dims N, G, H, H
        :param v_init: A tensor of initial vals for the values with dims N, G, Cv, H (optional)
        :param v_fixed: A tensor of fixed vals for the values with dims N, G, Cv, H (optional)
        :param zeta_prior_precision: A tensor of precisions for the Gaussian prior over zeta with dims N, G, C, H (optional)
        :param mu_prior_precision: A tensor of precisions for the Gaussian prior over mu with dims N, G, Cv, H (optional)
        :param q_pos_emb: A tensor of query positional embeddings with dims C, H, H
        :param zeta_pos_emb: A tensor of key positional embeddings with dims C, H, H
        :param v_pos_emb: A tensor of value positional embeddings with dims Cv, H, H
        :param nonzero_wts_mask: A boolean indexing tensor for setting weight matrix values to zero (where mask value is false) with dims H, H
        :return: Updated values with dims N, G, Cv, H if no position embedding (v_pos_emb=None) else N, G, 2*Cv, H
        """

        N, G, C_qk, H = q.shape
        C_v = mu.shape[-2]

        def update_weights():
            q_2 = torch.sum(q**2, dim=-2) #torch.sum(torch.square(q), dim=-2)
            zeta_2 = torch.sum(zeta**2, dim=-2) #torch.sum(torch.square(zeta), dim=-2)
            q_zeta = torch.einsum('bgci, bgcj->bgij', q, zeta)
            #q_m_zeta = q_2.unsqueeze(-1) + zeta_2.unsqueeze(-2) - 2 * q_zeta
            log_p_q_v = q_2.unsqueeze(-1) + zeta_2.unsqueeze(-2) - 2 * q_zeta
            if q_pos_emb is not None:
                q_pos_emb_2 = torch.sum(q_pos_emb**2, dim=0)
                q_q_pos_emb = torch.einsum('bgci, cij->bgij', q, q_pos_emb)
                #q_m_q_pos_emb = q_2.unsqueeze(-1) + q_pos_emb_2.unsqueeze(0).unsqueeze(0) - 2 * q_q_pos_emb
                #q_m_zeta += q_m_q_pos_emb
                log_p_q_v += q_2.unsqueeze(-1) + q_pos_emb_2.unsqueeze(0).unsqueeze(0) - 2 * q_q_pos_emb
            if zeta_pos_emb is not None:
                zeta_pos_emb_2 = torch.sum(zeta_pos_emb ** 2, dim=0).transpose(0, 1)
                zeta_zeta_pos_emb = torch.einsum('bgci, cij->bgij', zeta, zeta_pos_emb).transpose(2, 3)
                #zeta_m_zeta_pos_emb = zeta_2.unsqueeze(-2) + zeta_pos_emb_2.unsqueeze(0).unsqueeze(0) - 2 * zeta_zeta_pos_emb
                #q_m_zeta += zeta_m_zeta_pos_emb
                log_p_q_v += zeta_2.unsqueeze(-2) + zeta_pos_emb_2.unsqueeze(0).unsqueeze(0) - 2 * zeta_zeta_pos_emb
            if self.uniform_query_precision:
                #log_p_q = -0.5 * alpha * q_m_zeta
                log_p_q_v = -0.5 * alpha * log_p_q_v
            else:
                #log_p_q = -0.5 * alpha.unsqueeze(-2) * q_m_zeta
                log_p_q_v = -0.5 * alpha.unsqueeze(-2) * log_p_q_v

            #log_p_v = 0
            mu_2 = torch.sum(mu**2, dim=-2) #torch.sum(torch.square(mu), dim=-2)
            if v_init is not None:
                v_init_2 = torch.sum(v_init**2, dim=-2) #torch.sum(torch.square(v_init), dim=-2)
                v_init_mu = torch.einsum('bgci, bgcj->bgij', v_init, mu)
                #v_init_m_mu = v_init_2.unsqueeze(-1) + mu_2.unsqueeze(-2) - 2 * v_init_mu
                if self.uniform_value_precision:
                    #log_p_v = -0.5 * beta * v_init_m_mu
                    log_p_q_v += -0.5 * beta * (v_init_2.unsqueeze(-1) + mu_2.unsqueeze(-2) - 2 * v_init_mu)
                else:
                    #log_p_v = -0.5 * beta.unsqueeze(-2) * v_init_m_mu
                    log_p_q_v += -0.5 * beta.unsqueeze(-2) * (v_init_2.unsqueeze(-1) + mu_2.unsqueeze(-2) - 2 * v_init_mu)

            #log_pi = 0
            if pi is not None:
                #log_pi = torch.log(pi)
                log_p_q_v += torch.log(pi)
            elif self.magnitude_priors:
                if self.uniform_query_precision:
                    alpha_tensor = alpha
                else:
                    alpha_tensor = alpha.unsqueeze(-2)
                #log_pi += 0.5 * alpha_tensor * zeta_2.unsqueeze(-2)
                log_p_q_v += 0.5 * alpha_tensor * zeta_2.unsqueeze(-2)
                if q_pos_emb is not None:
                    #log_pi = log_pi.expand(-1, -1, H, -1).clone()
                    #log_pi += 0.5 * alpha_tensor * q_pos_emb_2.unsqueeze(0).unsqueeze(0)
                    log_p_q_v += 0.5 * alpha_tensor * q_pos_emb_2.unsqueeze(0).unsqueeze(0)
                if zeta_pos_emb is not None:
                    #log_pi += 0.5 * alpha_tensor * zeta_2.unsqueeze(-2)
                    #log_pi += 0.5 * alpha_tensor * zeta_pos_emb_2.unsqueeze(0).unsqueeze(0)
                    log_p_q_v += 0.5 * alpha_tensor * zeta_2.unsqueeze(-2)
                    log_p_q_v += 0.5 * alpha_tensor * zeta_pos_emb_2.unsqueeze(0).unsqueeze(0)
                if self.uniform_value_precision:
                    beta_tensor = beta
                else:
                    beta_tensor = beta.unsqueeze(-2)
                if v_pos_emb is not None:
                    mu_p_v_pos_emb = mu.unsqueeze(-2) + v_pos_emb.unsqueeze(0).unsqueeze(0)
                    mu_p_v_pos_emb_2 = torch.sum(mu_p_v_pos_emb**2, dim=-3)
                    #log_pi += 0.5 * beta_tensor * mu_p_v_pos_emb_2
                    log_p_q_v += 0.5 * beta_tensor * mu_p_v_pos_emb_2
                else:
                    #log_pi += 0.5 * beta_tensor * mu_2.unsqueeze(-2)
                    log_p_q_v += 0.5 * beta_tensor * mu_2.unsqueeze(-2)

            #log_p_q_v = log_pi + log_p_q + log_p_v
            # log_sum_exp trick to avoid numerical underflow
            m, idx = torch.max(log_p_q_v, dim=-1, keepdim=True)
            # Debugging
            """
            zeta_2_max, zeta_2_max_idx = torch.max(zeta_2, dim=-1, keepdim=True)
            log_pi_max, log_pi_max_idx = torch.max(log_pi, dim=-1, keepdim=True)
            """

            weights = torch.exp(log_p_q_v - m)
            if nonzero_wts_mask is not None:
                weights = weights * nonzero_wts_mask.unsqueeze(0).unsqueeze(0).float()
            sum_weights = torch.sum(weights, dim=-1, keepdim=True) + eps
            weights = weights.div(sum_weights)
            return weights

        weights = update_weights()

        if self.key_adaptation:
            # Online key adaptation
            for ka_iter in range(self.key_adaptation_iters):
                zeta_update = torch.einsum('bgij,bgci->bgcj', weights, q)
                sum_weights = torch.sum(weights, dim=-2, keepdim=True)
                if zeta_prior_precision is not None:
                    zeta = zeta_prior_precision * zeta + alpha * zeta_update
                    zeta = zeta.div(zeta_prior_precision + alpha * sum_weights)
                else:
                    zeta = zeta_update
                    zeta = zeta.div(sum_weights)
                weights = update_weights()

        wve = torch.zeros_like(mu).cuda() if torch.cuda.is_available() else torch.zeros_like(mu)
        if v_fixed is not None:
            # Online value belief propagation
            for vbp_iter in range(self.value_belief_propagation_iters):
                if torch.sum(v_fixed[:, :, -1, :]) > 0:
                    mu_update = torch.einsum('bgij,bgci->bgcj', weights, v_fixed[:, :, :C_v, :])
                    sum_weights = torch.einsum('bgij,bgi->bgj', weights, v_fixed[:, :, -1, :]).unsqueeze(-2) + eps
                    if mu_prior_precision is not None:
                        mu = mu_prior_precision * mu + beta * mu_update
                        mu = mu.div(mu_prior_precision + beta * sum_weights)
                    else:
                        mu = mu_update
                        mu = mu.div(sum_weights)
                    # Offset contributions from v_pos_emb with learnt parameters
                    if v_pos_emb is not None:
                        wve += torch.einsum('bgij,bgcj->bgci', weights, v_fixed[:, :, C_v:-1, :])
                    # Update weights
                    weights = update_weights()

        v_updated = torch.einsum('bgij,bgcj->bgci', weights, mu) # Should we force v_updated = v_fixed at specified locs?
        if v_pos_emb is not None:
            wve += torch.einsum('bgij,cij->bgci', weights, v_pos_emb)
            v_updated = torch.cat([v_updated, wve], dim=-1).view(N, G, C_v * 2, H)
        return v_updated
