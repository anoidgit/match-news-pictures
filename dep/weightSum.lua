local weightSum, parent = torch.class("nn.weightSum", "nn.Module")

function weightSum:__init()
	parent.__init(self)
	self.gradSeq = torch.Tensor()
	self.gradW = torch.Tensor()
end

function weightSum:updateOutput(input)
	local seqv, w = unpack(input)
	local stdv = seqv[1]
	if self.output:isSize(stdv:size()) then
		self.output:zero()
	else
		self.output:resizeAs(stdv):zero()
	end
	local bsize = w:size(2)
	local esize = stdv:size()
	for i = 1, seqv:size(1) do
		self.output:addcmul(w[i]:reshape(bsize,1):expand(esize), seqv[i])
	end
	self.wsum = w:sum()
	self.output:div(self.wsum)
	return self.output
end

function weightSum:updateGradInput(input, gradOutput)
	gradOutput:div(self.wsum)
	local seqv, w = unpack(input)
	if not self.gradSeq:isSize(seqv:size()) then
		self.gradSeq:resizeAs(seqv)
		self.gradW:resizeAs(w)
	end
	local bsize = w:size(2)
	local esize = gradOutput:size()
	for i = 1, seqv:size(1) do
		self.gradSeq[i]:cmul(gradOutput, w[i]:reshape(bsize,1):expand(esize))
		self.gradW[i] = torch.cmul(gradOutput, seqv[i]):sum()
	end
	self.gradInput = {self.gradSeq, self.gradW}
	return self.gradInput
end

function weightSum:clearState()
	parent.clearState()
	self.sum = nil
	self.gradSeq = nil
	self.gradW = nil
end
