import torch
import torch.nn as nn
import torch.nn.functional as F


class LayerNorm(nn.Module):
	
	def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last"):
		super().__init__()
		self.weight = nn.Parameter(torch.ones(normalized_shape))
		self.bias = nn.Parameter(torch.zeros(normalized_shape))
		self.eps = eps
		self.data_format = data_format
		if self.data_format not in ["channels_last", "channels_first"]:
			raise NotImplementedError
		self.normalized_shape = (normalized_shape,)
	
	def forward(self, x):
		if self.data_format == "channels_last":
			return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
		elif self.data_format == "channels_first":
			u = x.mean(1, keepdim=True)
			s = (x - u).pow(2).mean(1, keepdim=True)
			x = (x - u) / torch.sqrt(s + self.eps)
			x = self.weight[:, None, None] * x + self.bias[:, None, None]
			return x


class LayerNormFunction(torch.autograd.Function):
	
	@staticmethod
	def forward(ctx, x, weight, bias, eps):
		ctx.eps = eps
		N, C, H, W = x.size()
		mu = x.mean(1, keepdim=True)
		var = (x - mu).pow(2).mean(1, keepdim=True)
		y = (x - mu) / (var + eps).sqrt()
		ctx.save_for_backward(y, var, weight)
		y = weight.view(1, C, 1, 1) * y + bias.view(1, C, 1, 1)
		return y
	
	@staticmethod
	def backward(ctx, grad_output):
		eps = ctx.eps
		
		N, C, H, W = grad_output.size()
		y, var, weight = ctx.saved_variables
		g = grad_output * weight.view(1, C, 1, 1)
		mean_g = g.mean(dim=1, keepdim=True)
		
		mean_gy = (g * y).mean(dim=1, keepdim=True)
		gx = 1. / torch.sqrt(var + eps) * (g - y * mean_gy - mean_g)
		return gx, (grad_output * y).sum(dim=3).sum(dim=2).sum(dim=0), grad_output.sum(dim=3).sum(dim=2).sum(
			dim=0), None


class LayerNorm2d(nn.Module):
	
	def __init__(self, channels, eps=1e-6):
		super(LayerNorm2d, self).__init__()
		self.register_parameter('weight', nn.Parameter(torch.ones(channels)))
		self.register_parameter('bias', nn.Parameter(torch.zeros(channels)))
		self.eps = eps
	
	def forward(self, x):
		return LayerNormFunction.apply(x, self.weight, self.bias, self.eps)


class ResBlock(nn.Module):
	def __init__(self, channel):
		super(ResBlock, self).__init__()
		self.main = nn.Sequential(
			nn.Conv2d(channel, channel, kernel_size=3, padding=1, bias=True),
			nn.ReLU(inplace=True),
			nn.Conv2d(channel, channel, kernel_size=3, padding=1, bias=True),
		)
	
	def forward(self, x):
		return self.main(x) + x


class EResBlock(nn.Module):
	def __init__(self, channel):
		super().__init__()
		self.b1 = nn.Sequential(
			nn.Conv2d(channel, channel, kernel_size=1, bias=True),
			nn.ReLU(inplace=True),
			nn.Conv2d(channel, channel, kernel_size=1, bias=True),
		)
		self.b2 = nn.Sequential(
			nn.Conv2d(channel, channel, kernel_size=7, padding=3, groups=channel, bias=True),
			nn.ReLU(inplace=True),
			nn.Conv2d(channel, channel, kernel_size=7, padding=3, groups=channel, bias=True),
		)
	
	def forward(self, x):
		return self.b1(x) + self.b2(x) + x


class Rblock(nn.Module):
	def __init__(self, channel, t=2):
		super().__init__()
		self.t = t
		self.channel = channel
		self.norm = LayerNorm2d(channel)
		self.Wv = nn.Sequential(
			nn.Conv2d(channel, channel, kernel_size=5, padding=2, groups=channel, padding_mode='reflect'),
			nn.Conv2d(channel, channel, kernel_size=9, padding=12, groups=channel, dilation=3, padding_mode='reflect'),
			nn.Conv2d(channel, channel, 1),
		)
		
		self.Wg = nn.Sequential(
			nn.Conv2d(channel, channel, 1),
			# nn.Sigmoid()
		)
	
	def forward(self, x):
		x = self.norm(x)
		return self.Wv(x) * self.Wg(x)


class LKBlock(nn.Module):
	def __init__(self, channel):
		super().__init__()
		
		self.norm = LayerNorm2d(channel)
		self.w1 = nn.Parameter(torch.zeros((1, channel, 1, 1)), requires_grad=True)
		self.w2 = nn.Parameter(torch.zeros((1, channel, 1, 1)), requires_grad=True)
		
		self.R = Rblock(channel)
		self.proj = nn.Conv2d(channel, channel, 1)
		
		self.Wv1 = nn.Sequential(
			nn.Conv2d(channel, channel, kernel_size=3, padding=1, groups=channel, padding_mode='reflect'),
		)
		
		self.Wg1 = nn.Sequential(
			nn.Conv2d(channel, channel, 1),
			# nn.Sigmoid()
		)
		self.proj1 = nn.Conv2d(channel, channel, 1)
	
	def forward(self, inp):
		x = inp
		
		x = self.R(x)
		x1 = self.R(x)
		x = self.proj(x + x1)
		# x = self.proj(x )
		
		y = inp + x * self.w1
		
		x = self.norm(y)
		x = self.Wv1(x) * self.Wg1(x)
		x = self.proj1(x)
		
		return y + x * self.w2


class DeblurNet(nn.Module):
	
	def __init__(self, img_channel=3, channel=42, middle_nums=12, En_nums=[4, 4], De_nums=[4, 4]):
		super().__init__()
		
		self.intro = nn.Conv2d(img_channel, channel, kernel_size=3, padding=1, bias=True)
		self.ending = nn.Conv2d(channel, img_channel, kernel_size=3, padding=1, bias=True)
		
		self.En1 = nn.Sequential(
			*[LKBlock(channel) for _ in range(En_nums[0])]
		)
		self.En2 = nn.Sequential(
			*[LKBlock(channel * 2) for _ in range(En_nums[1])]
		)
		self.Down1 = nn.Sequential(
			nn.Conv2d(channel, 2 * channel, 2, 2)
		)
		self.Down2 = nn.Sequential(
			nn.Conv2d(channel * 2, 4 * channel, 2, 2)
		)
		
		self.middle = nn.Sequential(
			*[LKBlock(channel * 4) for _ in range(middle_nums)]
		)
		
		self.De1 = nn.Sequential(
			*[LKBlock(2 * channel) for _ in range(De_nums[0])]
		)
		self.De2 = nn.Sequential(
			*[LKBlock(channel) for _ in range(De_nums[1])]
		)
		self.Up1 = nn.Sequential(
			nn.Conv2d(4 * channel, 4 * channel * 2, 1, bias=False),
			nn.PixelShuffle(2)
		)
		self.Up2 = nn.Sequential(
			nn.Conv2d(2 * channel, 2 * channel * 2, 1, bias=False),
			nn.PixelShuffle(2)
		)
	
	def forward(self, blur):
		x = self.intro(blur)
		skip = []
		
		# 编码
		x = self.En1(x)
		skip.append(x)
		x = self.Down1(x)
		x = self.En2(x)
		skip.append(x)
		x = self.Down2(x)
		
		# Bottom
		x = self.middle(x)
		
		# 解码
		x = self.Up1(x)
		x = skip[-1] + x
		x = self.De1(x)
		x = self.Up2(x)
		x = skip[-2] + x
		x = self.De2(x)
		
		x = self.ending(x)
		x = x + blur
		
		return x, []

