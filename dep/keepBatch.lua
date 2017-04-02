local keepBatch, parent = torch.class('nn.keepBatch', 'nn.Module')

function keepBatch:updateOutput(input)
	if input:nDimension()==1 then
		self.output = input:reshape(1, input:size(1))
	else
		self.output = input
	end
	return self.output
end

function keepBatch:updateGradInput(input, gradOutput)
	if gradOutput:size(1)==1 then
		self.gradInput = gradOutput:reshape(gradOutput:size(2))
	else
		self.gradInput = gradOutput
	end
	return self.gradInput
end
