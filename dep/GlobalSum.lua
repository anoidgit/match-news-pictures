local GlobalSum, parent = torch.class("nn.GlobalSum", "nn.Module")

function GlobalSum:__init()
	parent.__init(self)
end

function GlobalSum:updateOutput(input)
	local osize = input:size()
	local seql = osize[1]
	osize[1] = 1
	self.output:resize(osize)
	self.output:sum(input, 1)
	self.output:div(seql)
	self.output = self.output:expandAs(input)
	return self.output
end

function GlobalSum:updateGradInput(input, gradOutput)
	local isize = input:size()
	local seql = isize[1]
	isize[1] = 1
	self.gradInput:resize(isize)
	self.gradInput:sum(gradOutput, 1)
	self.gradInput = self.gradInput:div(seql):expandAs(input)
	return self.gradInput
end
