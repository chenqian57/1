import torch
import torch.nn as nn
import torch.nn.functional as F

class LossModule(nn.Module):

	def __init__(self, opt):
		super(LossModule, self).__init__()

	def forward(self, scores):
		"""
		Loss based on Equation 6 from the TIRG paper,
		"Composing Text and Image for Image Retrieval - An Empirical Odyssey", CVPR19
		Nam Vo, Lu Jiang, Chen Sun, Kevin Murphy, Li-Jia Li, Li Fei-Fei, James Hays
		
		Args:
			scores: matrix of size (batch_size, batch_size), where coefficient
				(i,j) corresponds to the score between query (i) and target (j).
				Ground truth associations are represented along the diagonal.
		"""



		GT_labels = torch.arange(scores.shape[0]).long()
		GT_labels = torch.autograd.Variable(GT_labels)
		if torch.cuda.is_available():
			GT_labels = GT_labels.cuda()


		loss = F.cross_entropy(scores, GT_labels, reduction = 'mean')

		return loss

class ContrastiveLoss(nn.Module):
	"""
    Compute contrastive loss
    """
	def __init__(self, opt, margin=0):
		super(ContrastiveLoss, self).__init__()
		self.opt = opt
		self.embed_dim = opt.embed_dim
		self.margin = margin


	def forward(self, A_is_t, A_em_t, m, tr_m):

		batch_size = m.size(0)



		scores_is = (m.view(batch_size, 1, self.embed_dim) * A_is_t).sum(-1)
		scores_em = (m.view(batch_size, 1, self.embed_dim) * A_em_t).sum(-1)
		scores_is_trm = (tr_m.view(batch_size, 1, self.embed_dim) * A_is_t).sum(-1)
		scores_em_trm = (tr_m.view(batch_size, 1, self.embed_dim) * A_em_t).sum(-1)
		diagonal_is = scores_is.diag().view(batch_size, 1)
		diagonal_em = scores_em.diag().view(batch_size, 1)
		diagonal_is_trm = scores_is_trm.diag().view(batch_size, 1)
		diagonal_em_trm = scores_em_trm.diag().view(batch_size, 1)
		diagonal_is_all = (0.4 * diagonal_is + 0.6 * diagonal_is_trm)
		diagonal_em_all = (0.6 * diagonal_em_trm + 0.4 * diagonal_em)

		cost_s = (self.margin + diagonal_is_all - diagonal_em_all).clamp(min=0)
		return cost_s.sum(0)

class ContrastiveLoss_all(nn.Module):
	"""
    Compute contrastive loss
    """
	def __init__(self, opt, margin=0):
		super(ContrastiveLoss, self).__init__()
		self.opt = opt
		self.embed_dim = opt.embed_dim
		self.margin = margin


	def forward(self, A_is_t, A_is_t_14, A_is_t_28, A_em_t, A_em_t_14, A_em_t_28, m, tr_m):

		batch_size = m.size(0)



		scores_is = (m.view(batch_size, 1, self.embed_dim) * A_is_t).sum(-1)
		scores_is_14 = (m.view(batch_size, 1, self.embed_dim) * A_is_t_14).sum(-1)
		scores_is_28 = (m.view(batch_size, 1, self.embed_dim) * A_is_t_28).sum(-1)
		scores_is = scores_is + scores_is_14 + scores_is_28
		scores_em = (m.view(batch_size, 1, self.embed_dim) * A_em_t).sum(-1)
		scores_em_14 = (m.view(batch_size, 1, self.embed_dim) * A_em_t_14).sum(-1)
		scores_em_28 = (m.view(batch_size, 1, self.embed_dim) * A_em_t_28).sum(-1)
		scores_em = scores_em + scores_em_14 + scores_em_28
		scores_is_trm = (tr_m.view(batch_size, 1, self.embed_dim) * A_is_t).sum(-1)
		scores_is_trm_14 = (tr_m.view(batch_size, 1, self.embed_dim) * A_is_t_14).sum(-1)
		scores_is_trm_28 = (tr_m.view(batch_size, 1, self.embed_dim) * A_is_t_28).sum(-1)
		scores_is_trm = scores_is_trm + scores_is_trm_14 + scores_is_trm_28
		scores_em_trm = (tr_m.view(batch_size, 1, self.embed_dim) * A_em_t).sum(-1)
		scores_em_trm_14 = (tr_m.view(batch_size, 1, self.embed_dim) * A_em_t_14).sum(-1)
		scores_em_trm_28 = (tr_m.view(batch_size, 1, self.embed_dim) * A_em_t_28).sum(-1)
		scores_em_trm = scores_em_trm + scores_em_trm_14 + scores_em_trm_28
		diagonal_is = scores_is.diag().view(batch_size, 1)
		diagonal_em = scores_em.diag().view(batch_size, 1)
		diagonal_is_trm = scores_is_trm.diag().view(batch_size, 1)
		diagonal_em_trm = scores_em_trm.diag().view(batch_size, 1)
		diagonal_is_all = (0.4 * diagonal_is + 0.6 * diagonal_is_trm)
		diagonal_em_all = (0.6 * diagonal_em_trm + 0.4 * diagonal_em)

		cost_s = (self.margin + diagonal_is_all - diagonal_em_all).clamp(min=0)

		return cost_s.sum(0)