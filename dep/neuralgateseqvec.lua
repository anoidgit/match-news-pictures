require "dep.gateSum"

--this was needed for attention neural weight
require "dep.Attention"
--this was needed if you use sum vector rather than attention vector
--require "dep.GlobalSum"

require "nngraph"

function buildseqv(inivec,featsize,pdrop)
	local input = nn.vecLookup(inivec)()
	--local weight = nn.Bottle(nn.Linear(inivec:size(2), inivec:size(2), false))(input)
	local weight = nn.Bottle(nn.Linear(inivec:size(2) * 2, inivec:size(2), false))(nn.JoinTable(3)({input, nn.Attention()(input)}))
	--local weight = nn.Bottle(nn.Linear(inivec:size(2) * 2, inivec:size(2), false))(nn.JoinTable(3)({input, nn.GlobalSum()(input)}))
	local output = nn.Linear(inivec:size(2),featsize)(nn.ELU(nil,true)(nn.Dropout(pdrop or 0.5,nil,true)(nn.gateSum()({input, nn.Sigmoid()(weight)}))))
	return nn.gModule({input}, {output})
end

--[[function buildseqv(inivec,featsize,pdrop)
	local seqv=nn.Sequential()
		:add(nn.vecLookup(inivec))
		:add(nn.ConcatTable()
			:add(nn.Identity())
			:add(nn.Sequential()
				--:add(nn.Dropout(pdrop or 0.5))
				-- use linear or gru? gru need much more memory
				:add(nn.Bottle(nn.Linear(inivec:size(2), 1, false)))
				--:add(cudnn.GRU(inivec:size(2), hidsize , 1))
				--:add(nn.Bottle(nn.Linear(hidsize, 1)))
				:add(nn.reWShape())
				--:add(nn.Sigmoid())
				))
		:add(nn.weightSum())
		:add(nn.Dropout(pdrop or 0.5,nil,true))
		--:add(nn.Tanh())
		:add(nn.ELU(nil,true))
		:add(nn.Linear(inivec:size(2),featsize))
	return seqv
end]]
