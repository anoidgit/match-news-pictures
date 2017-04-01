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
	local usize = input[1]:size()
	if self.output:isSize(isize) then
		self.output:zero()
	else
		self.output:resize(isize):zero()
		self.w:resize(seql, seql, bsize)
	end
	for i = 1, seql do
		local center = input[i]
		local curw = self.w[i]
		for j = 1, seql do
			torch.sum(curw[j], torch.cmul(center, input[j]), 2)
		end
	end
	self.normw = self.module:updateOutput(self.w)
	for i = 1, seql do
		local curo = self.output[i]
		local curw = self.normw[i]
		for j = 1, seql do
			curo:addcmul(curw[j]:reshape(bsize, 1):expand(usize), input[j])
		end
	end
	return self.output
end

function Attention:updateGradInput(input, gradOutput)
	local isize = input:size()
	local seql = isize[1]
	local bsize = isize[2]
	local usize = input[1]:size()
	if self.gradInput:isSize(isize) then
		self.gradInput:zero()
		self.gradNormW:zero()
	else
		self.gradInput:resize(isize):zero()
		self.gradNormW:resize(seql, seql, bsize):zero()
	end
	for i = 1, seql do
		local curg = gradOutput[i]
		local curw = self.normw[i]
		local curgw = self.gradNormW[i]
		for j = 1, seql do
			self.gradInput[j]:addcmul(curg, curw[j]:reshape(bsize, 1):expand(usize))
			torch.sum(curgw[j], torch.cmul(curg, input[j]), 2)
		end
	end
	local gradW = self.module:updateGradInput(self.w, self.gradNormW)
	for i = 1, seql do
		local curgw = gradW[i]
		local center = input[i]
		local centerg = self.gradInput[i]
		for j = 1, seql do
			local curg = curgw[j]:reshape(bsize, 1):expand(usize)
			self.gradInput[j]:addcmul(curg, center)
			centerg:addcmul(curg, input[j])
		end
	end
	return self.gradInput
end

function Attention:clearState()
	parent.clearState()
	self.w = nil
	self.normw = nil
	self.gradNormW = nil
end
