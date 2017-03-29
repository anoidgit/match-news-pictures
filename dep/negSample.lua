local negSample, parent = torch.class('nn.negSample', 'nn.Module')

function negSample:__init()
	parent.__init(self)
	self.noise=torch.LongTensor()
end

function negSample:updateOutput(input)
	local isize=input:size(1)
	self.key=math.random(isize-1)
	self.noise:range(1,isize)
		:add(self.key)
		:remainder((isize))
	self.noise[self.noise:eq(0)]=isize
	self.output:index(input, 1, self.noise)
	return self.output
end

function negSample:updateGradInput(input, gradOutput)
	local isize=input:size(1)
	self.noise:range(1,isize)
		:add(isize-self.key)
		:remainder((isize))
	self.noise[self.noise:eq(0)]=isize
	self.gradInput:index(gradOutput, 1, self.noise)
	return self.gradInput
end

function negSample:clearState()
	self.key=nil
	self.noise = nil
	self.output = nil
	self.gradInput = nil
	return parent.clearState(self)
end
