local weightSum, parent = torch.class("nn.weightSum", "nn.Module")

function weightSum:__init()
	parent.__init(self)
	self.gradSeq = torch.Tensor()
	self.gradW = torch.Tensor()
end

function weightSum:updateOutput(input)
	local seqv, w = unpack(input)
	local isize = seqv:size()
	local seql = isize[1]
	local bsize = isize[2]
	local esize = seqv[1]:size()
	if not self.output:isSize(esize) then
		self.output:resize(esize)
	end
	self.output:sum(torch.cmul(w:reshape(seql, bsize, 1):expand(isize), seqv), 1)
	self.wsum = w:sum()
	self.output:div(self.wsum)
	self.output = self.output:reshape(esize)
	return self.output
end

function weightSum:updateGradInput(input, gradOutput)
	gradOutput:div(self.wsum)
	local seqv, w = unpack(input)
	local isize = seqv:size()
	if not self.gradSeq:isSize(isize) then
		self.gradSeq:resize(isize)
		self.gradW:resizeAs(w)
	end
	local seql = isize[1]
	local bsize = isize[2]
	local vsize = isize[3]
	local _g = gradOutput:reshape(1, bsize, vsize):expand(isize)
	self.gradSeq:cmul(_g, w:reshape(seql, bsize, 1):expand(isize))
	self.gradW:sum(torch.cmul(_g, seqv), 3)
	self.gradW = self.gradW:reshape(seql, bsize)
	self.gradInput = {self.gradSeq, self.gradW}
	return self.gradInput
end

function weightSum:clearState()
	self.wsum = nil
	self.gradSeq:resize(0)
	self.gradW:resize(0)
	return parent.clearState()
end
