local Attention, parent = torch.class("nn.Attention", "nn.Container")

function Attention:__init()
	parent.__init(self)
	self.module = nn.Sequential()
		:add(nn.Transpose({2,3}))
		:add(nn.SoftMax())
		:add(nn.Transpose({2,3}))
	self:add(self.module)
	self.w = torch.Tensor()
	self.gradNormW = torch.Tensor()
end

function Attention:updateOutput(input)
	local isize = input:size()
	local seql = isize[1]
	local bsize = isize[2]
	local vsize = isize[3]
	if self.output:isSize(isize) then
		self.output:zero()
	else
		self.output:resize(isize):zero()
		self.w:resize(seql, seql, bsize)
	end
	for i = 1, seql do
		self.w[i]:sum(torch.cmul(input[i]:reshape(1, bsize, vsize):expand(isize), input), 3)
	end
	self.normw = self.module:updateOutput(self.w)
	for i = 1, seql do
		self.output[i]:sum(torch.cmul(self.normw[i]:reshape(seql, bsize, 1):expand(isize), input), 1)
	end
	return self.output
end

function Attention:updateGradInput(input, gradOutput)
	local isize = input:size()
	local seql = isize[1]
	local bsize = isize[2]
	local vsize = isize[3]
	if self.gradInput:isSize(isize) then
		self.gradInput:zero()
		self.gradNormW:zero()
	else
		self.gradInput:resize(isize):zero()
		self.gradNormW:resize(seql, seql, bsize):zero()
	end
	for i = 1, seql do
		local curg = gradOutput[i]:reshape(1, bsize, vsize):expand(isize)
		self.gradInput:addcmul(curg, self.normw[i]:reshape(seql, bsize, 1):expand(isize))
		self.gradNormW[i]:sum(torch.cmul(curg, input), 3)
	end
	local gradW = self.module:updateGradInput(self.w, self.gradNormW)
	for i = 1, seql do
		local curgw = gradW[i]:reshape(seql, bsize, 1):expand(isize)
		self.gradInput:addcmul(curgw, input[i]:reshape(1, bsize, vsize):expand(isize))
		self.gradInput[i]:sum(torch.cmul(curgw, input), 1)
	end
	return self.gradInput
end

function Attention:clearState()
	self.w:resize(0)
	self.normw = nil
	self.gradNormW:resize(0)
	return parent.clearState()
end
