function buildseqv(inivec,featsize,pdrop,hsize)
	local hidsize = hsize or 2
	local seqv=nn.Sequential()
		:add(nn.ConcatTable()
			:add(nn.vecLookup(inivec))
			:add(nn.Sequential()
				-- or just a nn.Bottle(nn.Linear(inivec:size(2), 1))
				:add(cudnn.BGRU(inivec:size(2), hidsize , 1))
				:add(nn.Linear(2*hidsize, 1))
				:add(nn.Sigmoid())))
		:add(nn.weightSum())
		:add(nn.Dropout(pdrop or 0.5,nil,true))
		--:add(nn.Tanh())
		:add(nn.ELU(nil,true))
		:add(nn.Linear(inivec:size(2),featsize))
	return seqv
end
