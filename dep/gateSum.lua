local gateSum, parent = torch.class("nn.gateSum", "nn.Module")

function gateSum:__init()
	parent.__init(self)
	self.gradSeq = torch.Tensor()
	self.gradW = torch.Tensor()
end

function gateSum:updateOutput(input)
	local seqv, w = unpack(input)
	local esize = seqv[1]:size()
	if not self.output:isSize(esize) then
		self.output:resize(esize)
	end
	self.output:sum(torch.cmul(w, seqv), 1)
	self.output:div(seqv:size(1))
	self.output = self.output:reshape(esize)
	return self.output
end

function gateSum:updateGradInput(input, gradOutput)
	local seqv, w = unpack(input)
	local isize = seqv:size()
	if not self.gradSeq:isSize(isize) then
		self.gradSeq:resize(isize)
		self.gradW:resize(isize)
	end
	gradOutput:div(isize[1])
	local _g = gradOutput:reshape(1, isize[2], isize[3]):expand(isize)
	self.gradSeq:cmul(_g, w)
	self.gradW:cmul(_g, seqv)
	self.gradInput = {self.gradSeq, self.gradW}
	return self.gradInput
end

function gateSum:clearState()
	self.wsum = nil
	self.gradSeq:resize(0)
	self.gradW:resize(0)
	return parent.clearState()
end
