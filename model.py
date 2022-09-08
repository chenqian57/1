




import torch
import torch.nn as nn
from torch.autograd import Variable

from encoders import EncoderImage, EncoderText
from utils import params_require_grad, SimpleModule

class BaseModel(nn.Module):

	"""
	BaseModel for models to inherit from.
	Simply implement `compute_score` and `compute_score_broadcast`.
	"""

	def __init__(self, word2idx, opt):
		super(BaseModel, self).__init__()

		self.embed_dim = opt.embed_dim


		self.txt_enc = EncoderText(word2idx, opt)
		params_require_grad(self.txt_enc.embed, opt.txt_finetune)


		if opt.load_image_feature:
			self.img_enc = SimpleModule(opt.load_image_feature, self.embed_dim)

		else :
			self.img_enc = EncoderImage(opt)
			params_require_grad(self.img_enc.cnn, opt.img_finetune)



		self.temperature = nn.Parameter(torch.FloatTensor((opt.temperature,)))


	def get_image_embedding(self, images):
		return self.img_enc(Variable(images))


	def get_txt_embedding(self, sentences, lengths):
		return self.txt_enc(Variable(sentences), lengths)


















	def compute_score(self, r, r_14, r_28, m, t, t_14, t_28):
		raise NotImplementedError

	def compute_score_broadcast(self, r, r_14, r_28, m, t, t_14, t_28):
		raise NotImplementedError











	def forward(self, images_src, images_trg, sentences, lengths):
		"""
 		Returning a tensor of shape (batch_size), where coefficient (i) is the
		score between query (i) and target (i). 
		"""
		r, r_14, r_28 = self.get_image_embedding(images_src)
		t, t_14, t_28 = self.get_image_embedding(images_trg)
		m = self.get_txt_embedding(sentences, lengths)
		return self.compute_score(r, r_14, r_28, m, t, t_14, t_28)

	def forward_broadcast(self, images_src, images_trg, sentences, lengths):
		"""
		Returning a tensor of shape (batch_size, batch_size), corresponding to
		the score matrix where coefficient (i,j) is the score between query (i)
		and target (j).
		"""
		r, r_14, r_28 = self.get_image_embedding(images_src)
		m = self.get_txt_embedding(sentences, lengths)
		t, t_14, t_28 = self.get_image_embedding(images_trg)
		return self.compute_score_broadcast(r, r_14, r_28, m, t, t_14, t_28)

	def get_compatibility_from_embeddings_one_query_multiple_targets(self, r, r_14, r_28, m, t, t_14, t_28):
		"""
		Input:
		    - r: tensor of size (self.embed_dim), embedding of the query image.
		    - m: tensor of size (self.embed_dim), embedding of the query text.
		    - t: tensor of size (nb_imgs, self.embed_dim), embedding of the
		      candidate target images.

		Returns a tensor of size (1, nb_imgs) with the compatibility scores of
		each candidate target from t with regard to the provided query (r,m).
		"""
		return self.compute_score(r.view(1, -1), r_14.view(1, -1), r_28.view(1, -1), m.view(1, -1), t, t_14, t_28)