




import torch
import torch.nn as nn
import torch.nn.functional as F
from model import BaseModel
from utils import l2norm
from math import sqrt
class L2Module(nn.Module):

	def __init__(self):
		super(L2Module, self).__init__()

	def forward(self, x):
		x = l2norm(x)
		return x

def cosine_similarity(x1, x2, dim=1, eps=1e-8):
	"""Returns cosine similarity between x1 and x2, computed along dim."""
	w12 = torch.sum(x1 * x2, dim)
	w1 = torch.norm(x1, 2, dim)
	w2 = torch.norm(x2, 2, dim)
	return (w12 / (w1 * w2).clamp(min=eps)).squeeze()

class AttentionMechanism(nn.Module):


	def __init__(self, opt):
		super(AttentionMechanism, self).__init__()

		self.embed_dim = opt.embed_dim
		input_dim = self.embed_dim

		self.attention = nn.Sequential(
			nn.Linear(input_dim, self.embed_dim),
			nn.ReLU(),
			nn.Linear(self.embed_dim, self.embed_dim),
			nn.Softmax(dim=1)
		)

	def forward(self, x):
		return self.attention(x)







class Self_attention(nn.Module):




	def __init__(self, opt):
		super(Self_attention, self).__init__()

		self.input_dim = opt.embed_dim
		self.batch_size = opt.batch_size
		dim_k = self.input_dim
		dim_v = self.input_dim
		self.q = nn.Linear(self.input_dim, dim_k)
		self.k = nn.Linear(self.input_dim, dim_k)
		self.v = nn.Linear(self.input_dim, dim_v)
		self._norm_fact = 1 / sqrt(dim_k)


	def forward(self, x):

		Q = self.q(x)
		K = self.k(x)
		V = self.v(x)


		atten = nn.Softmax(dim=-1)(torch.bmm(Q.view(self.batch_size, self.input_dim, 1), K.view(self.batch_size, self.input_dim, 1).permute(0, 2, 1)))
		output = torch.bmm(atten, V.view(self.batch_size, self.input_dim, 1))
		return output.squeeze(-1)

class Self_attention_single(nn.Module):




	def __init__(self, opt):
		super(Self_attention_single, self).__init__()

		self.input_dim = opt.embed_dim

		dim_k = self.input_dim
		dim_v = self.input_dim
		self.q = nn.Linear(self.input_dim, dim_k)
		self.k = nn.Linear(self.input_dim, dim_k)
		self.v = nn.Linear(self.input_dim, dim_v)
		self._norm_fact = 1 / sqrt(dim_k)


	def forward(self, x):

		Q = self.q(x)
		K = self.k(x)
		V = self.v(x)


		atten = nn.Softmax(dim=-1)(torch.mm(Q.view(self.input_dim, 1), K.view(self.input_dim, 1).permute(1, 0)))
		output = torch.mm(atten, V.view(self.input_dim, 1))
		return output.transpose(0, 1).contiguous()

class Self_attention_7(nn.Module):




	def __init__(self, opt):
		super(Self_attention_7, self).__init__()

		self.input_dim = opt.embed_dim
		self.batch_size = opt.batch_size
		dim_k = self.input_dim
		dim_v = self.input_dim
		self.q = nn.Linear(self.input_dim, dim_k)
		self.k = nn.Linear(self.input_dim, dim_k)
		self.v = nn.Linear(self.input_dim, dim_v)
		self._norm_fact = 1 / sqrt(dim_k)


	def forward(self, x):

		Q = self.q(x)
		K = self.k(x)
		V = self.v(x)


		atten = nn.Softmax(dim=-1)(torch.bmm(Q, K.permute(0, 2, 1)))
		output = torch.bmm(atten, V)
		return output

class Self_attention_single_7(nn.Module):




	def __init__(self, opt):
		super(Self_attention_single_7, self).__init__()

		self.input_dim = opt.embed_dim

		dim_k = self.input_dim
		dim_v = self.input_dim
		self.q = nn.Linear(self.input_dim, dim_k)
		self.k = nn.Linear(self.input_dim, dim_k)
		self.v = nn.Linear(self.input_dim, dim_v)
		self._norm_fact = 1 / sqrt(dim_k)


	def forward(self, x):

		Q = self.q(x)
		K = self.k(x)
		V = self.v(x)


		atten = nn.Softmax(dim=-1)(torch.bmm(Q, K.permute(0, 2, 1)))
		output = torch.bmm(atten, V)
		return output

def cross_attention(cap_emb, img_emb, smooth, eps=1e-8):
	"""
    cap_emb: (batch_size, seq_maxlen, emb_dim) (32, maxlen(1), 1024/2)
    img_emb: (batch_size, h*w, channe(emb_dim)) (32, 7*7, 2048/4)
    output: attention_t2i(batchsize, seqlen, h*w)
    smooth: 平滑参数
    """
	n_cap, caplen = cap_emb.size(0), cap_emb(1)
	n_img, imghw = img_emb.size(0), img_emb(1)

	cap_emb_T = torch.transpose(cap_emb, 1, 2).contiguous()
	attn_weight = torch.bmm(img_emb, cap_emb_T)
	attn_weight = nn.LeakyReLU(0.1)(attn_weight)
	attn_weight = l2norm(attn_weight, -1)
	attn_weight = torch.transpose(attn_weight, 1, 2).contiguous()
	attn_weight = F.softmax(attn_weight*smooth, -1)
	attn_weight = torch.transpose(attn_weight, 1, 2).contiguous()
	'''
	img_emb_T = torch.transpose(img_emb, 1, 2).contiguous() 
	weighted_imgemb = torch.bmm(img_emb_T, attn_weight) 
	weighted_contextT = torch.transpose(weighted_imgemb, 1, 2).contiguous() 
	weighted_contextT = l2norm(weighted_contextT, -1)
	'''
	return attn_weight
class cross_attentioni2t_single(nn.Module):




	def __init__(self, opt):
		super(cross_attentioni2t_single, self).__init__()

		self.input_dim = opt.embed_dim

		dim_k = self.input_dim
		dim_v = self.input_dim
		self.q = nn.Linear(self.input_dim, dim_k)
		self.k = nn.Linear(self.input_dim, dim_k)
		self.v = nn.Linear(self.input_dim, dim_v)
		self._norm_fact = 1 / sqrt(dim_k)


	def forward(self, cap_emb, img_emb):


		cap_emb = cap_emb.unsqueeze(0)
		cap_emb = cap_emb.repeat(img_emb.size(0), 1, 1)
		Q = self.q(cap_emb)
		K = self.k(img_emb)
		V = self.v(img_emb)





		atten = nn.Softmax(dim=-1)(torch.bmm(Q, K.permute(0, 2, 1))) * self._norm_fact
		output = torch.bmm(atten, V)

		return output

class cross_attentioni2t_b(nn.Module):





	def __init__(self, opt):
		super(cross_attentioni2t_b, self).__init__()

		self.input_dim = opt.embed_dim
		self.batch_size = opt.batch_size
		dim_k = self.input_dim
		dim_v = self.input_dim
		self.q = nn.Linear(self.input_dim, dim_k)
		self.k = nn.Linear(self.input_dim, dim_k)
		self.v = nn.Linear(self.input_dim, dim_v)
		self._norm_fact = 1 / sqrt(dim_k)


	def forward(self, cap_emb, img_emb):

		cap_emb = cap_emb.unsqueeze(1)
		Q = self.q(cap_emb)
		K = self.k(img_emb)
		V = self.v(img_emb)


		atten = nn.Softmax(dim=-1)(torch.bmm(Q,  K.permute(0, 2, 1))) * self._norm_fact
		output = torch.bmm(atten, V)

		return output

class cross_attention_single(nn.Module):




	def __init__(self, opt):
		super(cross_attention_single, self).__init__()

		self.input_dim = opt.embed_dim

		dim_k = self.input_dim
		dim_v = self.input_dim
		self.q = nn.Linear(self.input_dim, dim_k)
		self.k = nn.Linear(self.input_dim, dim_k)
		self.v = nn.Linear(self.input_dim, dim_v)
		self._norm_fact = 1 / sqrt(dim_k)


	def forward(self, cap_emb, img_emb):


		cap_emb = cap_emb.unsqueeze(0)
		cap_emb = cap_emb.repeat(img_emb.size(0), 1, 1)
		Q = self.q(img_emb)
		K = self.k(cap_emb)
		V = self.v(cap_emb)





		atten = nn.Softmax(dim=1)(nn.ReLU(torch.bmm(Q, K.view(img_emb.size(0), 1, self.input_dim).permute(0, 2, 1)))) * self._norm_fact
		output = torch.bmm(atten, V.view(img_emb.size(0), 1, self.input_dim))
		return output

class cross_attention_b(nn.Module):





	def __init__(self, opt):
		super(cross_attention_b, self).__init__()

		self.input_dim = opt.embed_dim
		self.batch_size = opt.batch_size
		dim_k = self.input_dim
		dim_v = self.input_dim
		self.q = nn.Linear(self.input_dim, dim_k)
		self.k = nn.Linear(self.input_dim, dim_k)
		self.v = nn.Linear(self.input_dim, dim_v)
		self._norm_fact = 1 / sqrt(dim_k)


	def forward(self, cap_emb, img_emb):
		Q = self.q(img_emb)
		K = self.k(cap_emb)
		V = self.v(cap_emb)


		atten = nn.Softmax(dim=1)(nn.ReLU(torch.bmm(Q,  K.view(self.batch_size, 1, self.input_dim).permute(0, 2, 1)))) * self._norm_fact
		output = torch.bmm(atten, V.view(self.batch_size, 1, self.input_dim))
		return output


class MANME(BaseModel):


	def __init__(self, word2idx, opt):
		super(MANME, self).__init__(word2idx, opt)


		self.Transform_m = nn.Sequential(nn.Linear(self.embed_dim, self.embed_dim), L2Module())

		self.Transform_attention = nn.Sequential(nn.Linear(self.embed_dim, self.embed_dim))
		self.Transform_m_first = Self_attention(opt)
		self.Transform_m_first_single = Self_attention_single(opt)
		self.Transform_m_first_single_7 = Self_attention_single_7(opt)
		self.Transform_attention_7 = Self_attention_7(opt)
		self.Attention_EM = AttentionMechanism(opt)
		self.Attention_IS = AttentionMechanism(opt)
		self.Attention_IS_first_single = cross_attention_single
		self.Transform_m_attention_double_single = nn.Sequential(self.Transform_m_first_single, nn.Linear(self.embed_dim, self.embed_dim)
														  , self.Transform_m_first_single, nn.Linear(self.embed_dim, self.embed_dim))
		self.Transform_m_attention_double = nn.Sequential(self.Transform_m_first, nn.Linear(self.embed_dim, self.embed_dim)
																 , self.Transform_m_first, nn.Linear(self.embed_dim, self.embed_dim))
		self.cross_attention_sig = cross_attention_single(opt)
		self.cross_attention_ba = cross_attention_b(opt)
		self.cross_attentioni2t_sig = cross_attentioni2t_single(opt)
		self.cross_attentioni2t_ba = cross_attentioni2t_b(opt)


		self.model_version = opt.model_version
		if self.model_version == "MANME":
			self.compute_score = self.compute_score_artemis
			self.compute_score_broadcast = self.compute_score_broadcast_artemis



		self.gradcam = opt.gradcam
		self.hold_results = dict()


















	def apply_attention(self, a, x):
		return l2norm(a * x)
	
	def compute_score_artemis(self, r, r_14, r_28, m, t, t_14, t_28, store_intermediary=False):
		EM, EM_14, EM_28 = self.compute_score_EM(r, r_14, r_28, m, t, t_14, t_28, store_intermediary)

		IS, IS_14, IS_28 = self.compute_score_IS(r, r_14, r_28, m, t, t_14, t_28, store_intermediary)
		if store_intermediary:
			self.hold_results["EM"] = EM
			self.hold_results["IS"] = IS

		return EM + IS, EM_14 + IS_14, EM_28 + IS_28
	def compute_score_broadcast_artemis(self, r, r_14, r_28, m, t, t_14, t_28):
		EM, EM_14, EM_28, A_EM_all_t, A_EM_all_t_14, A_EM_all_t_28, Tr_m = self.compute_score_broadcast_EM(r, r_14, r_28, m, t, t_14, t_28)
		IS, IS_14, IS_28, A_IS_all_t, A_IS_all_t_14, A_IS_all_t_28 = self.compute_score_broadcast_IS(r, r_14, r_28, m, t, t_14, t_28)
		return EM + IS, EM_14 + IS_14, EM_28 + IS_28, A_IS_all_t, A_IS_all_t_14, A_IS_all_t_28, A_EM_all_t, A_EM_all_t_14, A_EM_all_t_28, m, Tr_m



	def compute_score_EM(self, r, r_14, r_28, m, t, t_14, t_28, store_intermediary=False):
		Tr_m = self.Transform_m(m)


		A_EM_t = self.apply_attention(self.Attention_EM(m), t)
		A_EM_t_14 = self.apply_attention(self.Attention_EM(m), t_14)
		A_EM_t_28 = self.apply_attention(self.Attention_EM(m), t_28)
		if store_intermediary:
			self.hold_results["Tr_m"] = Tr_m
			self.hold_results["A_EM_t"] = A_EM_t
		return (Tr_m * A_EM_t).sum(-1), (Tr_m * A_EM_t_14).sum(-1), (Tr_m * A_EM_t_28).sum(-1)
	def compute_score_broadcast_EM(self, r, r_14, r_28, m, t, t_14, t_28):
		batch_size = r.size(0)

		A_EM = self.Attention_EM(m)

		Tr_m = self.Transform_m(m)

		A_EM_all_t = self.apply_attention(A_EM.view(batch_size, 1, self.embed_dim), t.view(1, batch_size, self.embed_dim))
		A_EM_all_t_14 = self.apply_attention(A_EM.view(batch_size, 1, self.embed_dim), t_14.view(1, batch_size, self.embed_dim))
		A_EM_all_t_28 = self.apply_attention(A_EM.view(batch_size, 1, self.embed_dim), t_28.view(1, batch_size, self.embed_dim))
		EM_score = (Tr_m.view(batch_size, 1, self.embed_dim) * A_EM_all_t).sum(-1)
		EM_score_14 = (Tr_m.view(batch_size, 1, self.embed_dim) * A_EM_all_t_14).sum(-1)
		EM_score_28 = (Tr_m.view(batch_size, 1, self.embed_dim) * A_EM_all_t_28).sum(-1)


		return EM_score, EM_score_14, EM_score_28, A_EM_all_t, A_EM_all_t_14, A_EM_all_t_28, Tr_m





	def compute_score_IS(self, r, r_14, r_28, m, t, t_14, t_28, store_intermediary=False):


		A_IS_r = self.apply_attention(self.Attention_IS(m), r)
		A_IS_t = self.apply_attention(self.Attention_IS(m), t)
		A_IS_r_14 = self.apply_attention(self.Attention_IS(m), r_14)
		A_IS_t_14 = self.apply_attention(self.Attention_IS(m), t_14)
		A_IS_r_28 = self.apply_attention(self.Attention_IS(m), r_28)
		A_IS_t_28 = self.apply_attention(self.Attention_IS(m), t_28)






		if store_intermediary:
			self.hold_results["A_IS_r"] = A_IS_r
			self.hold_results["A_IS_t"] = A_IS_t
		return (A_IS_r * A_IS_t).sum(-1), (A_IS_r_14 * A_IS_t_14).sum(-1), (A_IS_r_28 * A_IS_t_28).sum(-1)



	def compute_score_broadcast_IS(self, r, r_14, r_28, m, t, t_14, t_28):
		batch_size = r.size(0)

		A_IS = self.Attention_IS(m)





		A_IS_r = self.apply_attention(A_IS, r)


		A_IS_all_t = self.apply_attention(A_IS.view(batch_size, 1, self.embed_dim), t.view(1, batch_size, self.embed_dim))
		A_IS_r_14 = self.apply_attention(A_IS, r_14)


		A_IS_all_t_14 = self.apply_attention(A_IS.view(batch_size, 1, self.embed_dim), t_14.view(1, batch_size, self.embed_dim))
		A_IS_r_28 = self.apply_attention(A_IS, r_28)


		A_IS_all_t_28 = self.apply_attention(A_IS.view(batch_size, 1, self.embed_dim), t_28.view(1, batch_size, self.embed_dim))
		IS_score = (A_IS_r.view(batch_size, 1, self.embed_dim) * A_IS_all_t).sum(-1)
		IS_score_14 = (A_IS_r_14.view(batch_size, 1, self.embed_dim) * A_IS_all_t_14).sum(-1)
		IS_score_28 = (A_IS_r_28.view(batch_size, 1, self.embed_dim) * A_IS_all_t_28).sum(-1)
		return IS_score, IS_score_14, IS_score_28, A_IS_all_t, A_IS_all_t_14, A_IS_all_t_28

	def compute_score_arithmetic(self, r, m, t, store_intermediary=False):
		return (l2norm(r + m) * t).sum(-1)
	def compute_score_broadcast_arithmetic(self, r, m, t):
		return (l2norm(r + m)).mm(t.t())

	def compute_score_crossmodal(self, r, m, t, store_intermediary=False):
		return (m * t).sum(-1)
	def compute_score_broadcast_crossmodal(self, r, m, t):
		return m.mm(t.t())

	def compute_score_visualsearch(self, r, m, t, store_intermediary=False):
		return (r * t).sum(-1)
	def compute_score_broadcast_visualsearch(self, r, m, t):
		return r.mm(t.t())






	def forward_save_intermediary(self, images_src, images_trg, sentences, lengths):


		self.hold_results.clear()


		r = self.get_image_embedding(images_src)
		if self.gradcam:
			self.hold_results["r_activation"] = self.img_enc.get_activation()
		t = self.get_image_embedding(images_trg)
		if self.gradcam:
			self.hold_results["t_activation"] = self.img_enc.get_activation()
		m = self.get_txt_embedding(sentences, lengths)

		return self.compute_score(r, m, t, store_intermediary=True)