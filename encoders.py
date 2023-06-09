




import os
import torch
import torch.nn as nn
import torchtext
import torchvision
from torch.nn.utils.rnn import pack_padded_sequence
from torch.nn.utils.rnn import pad_packed_sequence
from resnet import resnet50, resnet18

from utils import l2norm
from config import TORCH_HOME, GLOVE_DIR

os.environ['TORCH_HOME'] = TORCH_HOME


def get_cnn(arch):
	return torchvision.models.__dict__[arch](pretrained=True)


class GeneralizedMeanPooling(nn.Module):
	"""
	Applies a 2D power-average adaptive pooling over an input signal composed of several input planes.

	The function computed is: :math:`f(X) = pow(sum(pow(X, p)), 1/p)`

		- At p = infinity, one gets Max Pooling
		- At p = 1, one gets Average Pooling

	The output is of size H x W, for any input size.
	The number of output features is equal to the number of input planes.

	Args:
		output_size: the target output size of the image of the form H x W.
					 Can be a tuple (H, W) or a single H for a square image H x H
					 H and W can be either a ``int``, or ``None`` which means the size will
					 be the same as that of the input.
	"""

	def __init__(self, norm, output_size=1, eps=1e-6):
		super(GeneralizedMeanPooling, self).__init__()
		assert norm > 0
		self.p = float(norm)
		self.output_size = output_size
		self.eps = eps

	def forward(self, x):
		x = x.clamp(min=self.eps).pow(self.p)
		return torch.nn.functional.adaptive_avg_pool2d(x, self.output_size).pow(1. / self.p)

	def __repr__(self):
		return self.__class__.__name__ + '(' \
			+ str(self.p) + ', ' \
			+ 'output_size=' + str(self.output_size) + ')'


class EncoderImage(nn.Module):

	def __init__(self, opt):
		super(EncoderImage, self).__init__()


		embed_dim = opt.embed_dim
		self.gradcam = opt.gradcam



		self.cnn = resnet50(pretrained=True)
		self.cnn_dim = self.cnn.fc.in_features
		self.pool_dim = self.cnn_dim



		self.cnn.avgpool = nn.Sequential()
		self.cnn.fc = nn.Sequential()
		self.gemp = GeneralizedMeanPooling(norm=3)
		self.fc = nn.Linear(self.pool_dim, embed_dim)
		self.pool_dim_14 = 1024
		self.pool_dim_28 = 512
		self.fc_512 = nn.Linear(512, embed_dim)
		self.fc_1024 = nn.Linear(1024, embed_dim)

		if opt.gradcam :

			self.activations = None

			self.gradients = None

	@property
	def dtype(self):
		return self.cnn.conv1.weight.dtype

	def forward(self, images):




		out_7x7, out_14, out_28 = self.cnn(images)

		out_7x7 = out_7x7.view(-1, self.cnn_dim, 7, 7).type(self.dtype)
		out_14 = out_14.view(-1, 1024, 14, 14).type(self.dtype)
		out_28 = out_28.view(-1, 512, 28, 28).type(self.dtype)

		if self.gradcam:
			out_7x7.requires_grad_(True)

			self.register_activations(out_7x7)

			h = out_7x7.register_hook(self.activations_hook)


		out = self.gemp(out_7x7).view(-1, self.pool_dim)
		out = self.fc(out)
		out = l2norm(out)

		output_14 = self.gemp(out_14).view(-1, self.pool_dim_14)
		out_14 = self.fc_1024(output_14)
		out_14 = l2norm(out_14)
		output_28 = self.gemp(out_28).view(-1, self.pool_dim_28)
		out_28 = self.fc_512(output_28)
		out_28 = l2norm(out_28)

		return out, out_14, out_28



	def activations_hook(self, grad):
		""" hook for the gradients of the activations """
		self.gradients = grad

	def register_activations(self, activations):
		self.activations = activations

	def get_gradient(self):
		""" gradient extraction """
		return self.gradients

	def get_activation(self):
		""" activation extraction """
		return self.activations


class EncoderText(nn.Module):

	def __init__(self, word2idx, opt):
		super(EncoderText, self).__init__()

		wemb_type, word_dim, embed_dim = \
			opt.wemb_type, opt.word_dim, opt.embed_dim

		self.txt_enc_type = opt.txt_enc_type
		self.embed_dim = embed_dim


		self.embed = nn.Embedding(len(word2idx), word_dim)


		if self.txt_enc_type == "bigru":
			self.sent_enc = nn.GRU(word_dim, embed_dim//2, bidirectional=True, batch_first=True)
			self.forward = self.forward_bigru
		elif self.txt_enc_type == "lstm":
			self.lstm_hidden_dim = opt.lstm_hidden_dim
			self.sent_enc = nn.Sequential(
				nn.LSTM(word_dim, self.lstm_hidden_dim),
				nn.Dropout(p=0.1),
				nn.Linear(self.lstm_hidden_dim, embed_dim),
			)
			self.forward = self.forward_lstm

		self.init_weights(wemb_type, word2idx, word_dim)


	def init_weights(self, wemb_type, word2idx, word_dim):
		if wemb_type is None:
			print("Word embeddings randomly initialized with xavier")
			nn.init.xavier_uniform_(self.embed.weight)
		else:

			if 'glove' == wemb_type.lower():
				wemb = torchtext.vocab.GloVe(cache=GLOVE_DIR)
			else:
				raise Exception('Unknown word embedding type: {}'.format(wemb_type))
			assert wemb.vectors.shape[1] == word_dim


			missing_words = []
			for word, idx in word2idx.items():
				if word in wemb.stoi:
					self.embed.weight.data[idx] = wemb.vectors[wemb.stoi[word]]
				else:
					missing_words.append(word)
			print('Words: {}/{} found in vocabulary; {} words missing'.format(
				len(word2idx)-len(missing_words), len(word2idx), len(missing_words)))

	@property
	def dtype(self):
		return self.embed.weight.data.dtype

	@property
	def device(self):
		return self.embed.weight.data.device

	def forward_bigru(self, x, lengths):


		wemb_out = self.embed(x)


		lengths = lengths.cpu()


		packed = pack_padded_sequence(wemb_out, lengths, batch_first=True)
		if torch.cuda.device_count() > 1:
			self.sent_enc.flatten_parameters()

		_, rnn_out = self.sent_enc(packed)

		rnn_out = rnn_out.permute(1, 0, 2).contiguous().view(-1, self.embed_dim)

		out = l2norm(rnn_out)
		return out

	def forward_bigru_output(self, captions, lengths):

		cap_emb = self.emb(captions)
		cap_emb = self.dropout(cap_emb)

		pack_caption = pack_padded_sequence(cap_emb, lengths, batch_first=True)
		out, _ = self.rnn(pack_caption)
		cap_emb, _ = pad_packed_sequence(out, batch_first=True)
		if self.use_bi_gru:
			cap_emb = cap_emb[:, :, :cap_emb.size(2) / 2] + cap_emb[:, :, cap_emb.size(2) / 2:]
		cap_emb = l2norm(cap_emb, dim=-1)
		return cap_emb


	def forward_lstm(self, x, lengths):


		wemb_out = self.embed(x)
		wemb_out = wemb_out.permute(1, 0, 2)


		batch_size = wemb_out.size(1)
		first_hidden = (torch.zeros(1, batch_size, self.lstm_hidden_dim),
						torch.zeros(1, batch_size, self.lstm_hidden_dim))
		if torch.cuda.is_available():
			first_hidden = (first_hidden[0].cuda(), first_hidden[1].cuda())
		lstm_output, last_hidden = self.sent_enc[0](wemb_out, first_hidden)


		text_features = []
		for i in range(batch_size):
			text_features.append(lstm_output[:, i, :].max(0)[0])
		text_features = torch.stack(text_features)


		out = self.sent_enc[1:](text_features)
		out = l2norm(out)
		return out

	def forward_lstm_output(self, x, lengths):


		wemb_out = self.embed(x)
		wemb_out = wemb_out.permute(1, 0, 2)


		batch_size = wemb_out.size(1)
		first_hidden = (torch.zeros(1, batch_size, self.lstm_hidden_dim),
						torch.zeros(1, batch_size, self.lstm_hidden_dim))
		if torch.cuda.is_available():
			first_hidden = (first_hidden[0].cuda(), first_hidden[1].cuda())
		lstm_output, last_hidden = self.sent_enc[0](wemb_out, first_hidden)


		text_features = []
		for i in range(batch_size):
			text_features.append(lstm_output[:, i, :].max(0)[0])
		text_features = torch.stack(text_features)


		out = self.sent_enc[1:](text_features)
		out = l2norm(out)
		return out