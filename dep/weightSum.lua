local weightSum, parent = torch.class("nn.weightSum", "nn.Module")

function weightSum:__init()
	parent.__init(self)
	self.gradSeq = torch.Tensor()
	self.gradW = torch.Tensor()
end

function weightSum:updateOutput(input)
	local seqv, w = unpack(input)
	local stdv = seqv[1]
	if self.output:isSize(stdv) then
		self.output:zero()
	else
		self.output:resizeAs(stdv):zero()
	end
	for i = 1, seqv:size(1) do
		self.output:addcmul(w[i], seqv[i])
	end
	self.wsum = w:sum()
	self.output:div(self.wsum)
	return self.output
end

function weightSum:updateGradInput(input, gradOutput)
	gradOutput:div(self.wsum)
	local seqv, w = unpack(input)
	if ~self.gradSeq:isSize(seqv) then
		self.gradSeq:resizeAs(seqv):zero()
		self.gradW:resizeAs(w):zero()
	end
	for i = 1, seqv:size(1) do
		local _curGrad = gradOutput[i]
		self.gradSeq[i]:mul(_curGrad, w[i])
		self.gradW[i] = torch.cmul(_curGrad, seqv[i]):sum()
	end
	self.gradInput = {self.gradSeq, self.gradW}
end

function weightSum:clearState()
	parent.clearState()
	self.sum = nil
	self.gradSeq = nil
	self.gradW = nil
end