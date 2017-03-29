function buildseqv(inivec,featsize,pdrop)
	local seqv=nn.Sequential()
		:add(nn.vecLookup(inivec))
		:add(nn.Sum(2,3,true))
		:add(nn.Dropout(pdrop or 0.5,nil,true))
		--:add(nn.Tanh())
		:add(nn.ELU(nil,true))
		:add(nn.Linear(inivec:size(2),featsize))
	return seqv
end
