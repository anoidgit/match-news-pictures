local reWShape, parent = torch.class("nn.reWShape", "nn.Module")

function reWShape:__init()
	parent.__init(self)
end

function reWShape:updateOutput(input)
	self.output = input:reshape(input:size(1), input:size(2))
	return self.output
end

function reWShape:updateGradInput(input, gradOutput)
	self.gradInput = gradOutput:reshape(input:size())
	return self.gradInput
end
