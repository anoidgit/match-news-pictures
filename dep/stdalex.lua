function buildalex(featsize)
	local SpatialConvolution = cudnn.SpatialConvolution
	local SpatialMaxPooling = cudnn.SpatialMaxPooling

	local alex = nn.Sequential()
		:add(SpatialConvolution(3,64,11,11,4,4,2,2))		 -- 224 -> 55
		:add(SpatialMaxPooling(3,3,2,2))						 -- 55 ->  27
		:add(nn.ELU(nil,true))
		:add(SpatialConvolution(64,192,5,5,1,1,2,2))		 --  27 -> 27
		:add(SpatialMaxPooling(3,3,2,2))						 --  27 ->  13
		:add(nn.ELU(nil,true))
		:add(SpatialConvolution(192,384,3,3,1,1,1,1))		--  13 ->  13
		:add(nn.ELU(nil,true))
		:add(SpatialConvolution(384,256,3,3,1,1,1,1))		--  13 ->  13
		:add(nn.ELU(nil,true))
		:add(SpatialConvolution(256,256,3,3,1,1,1,1))		--  13 ->  13
		:add(SpatialMaxPooling(3,3,2,2))						 -- 13 -> 6

		:add(nn.View(256*6*6))
		:add(nn.Dropout(0.5))
		:add(nn.ELU(nil,true))
		:add(nn.Linear(256*6*6, 4096))
		:add(nn.Dropout(0.5))
		:add(nn.ELU(nil,true))
		:add(nn.Linear(4096, 4096))
		:add(nn.ELU(nil,true))

		--:add(nn.Dropout(0.5))
		:add(nn.Linear(4096, featsize))
		:add(nn.keepBatch())

	local function ConvInit(name)
		for k,v in pairs(alex:findModules(name)) do
			local n = v.kW*v.kH*v.nOutputPlane
			v.weight:normal(0,math.sqrt(2/n))
			v.bias = nil
			v.gradBias = nil
		end
	end

	ConvInit('cudnn.SpatialConvolution')
	for k,v in pairs(alex:findModules('nn.Linear')) do
		v.bias:zero()
	end

	alex:get(1).gradInput = nil

	return alex
end
