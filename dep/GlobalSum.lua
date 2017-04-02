local GlobalSum, parent = torch.class("nn.GlobalSum", "nn.Module")

function GlobalSum:__init()
	parent.__init(self)
end

function GlobalSum:updateOutput(input)
	local osize = input:size()
	local seql = osize[1]
	osize[1] = 1
	self.output:resize(osize)
	torch.sum(self.output, input, 1)
	self.output:div(seql)
	self.output = self.output:expandAs(input)
	return self.output
end

function GlobalSum:updateGradInput(input, gradOutput)
	local isize = input:size()
	local seql = isize[1]
	isize[1] = 1
	self.gradInput:resize(isize)
	torch.sum(self.gradInput, gradOutput, 1)
	self.gradInput = self.gradInput:div(seql):expandAs(input)
	return self.gradInput
end
