function buildalex(featsize)
	local SpatialConvolution = cudnn.SpatialConvolution
	local SpatialMaxPooling = cudnn.SpatialMaxPooling

	-- from https://code.google.com/p/cuda-convnet2/source/browse/layers/layers-imagenet-1gpu.cfg
	-- this is Net that was presented in the One Weird Trick paper. http://arxiv.org/abs/1404.5997
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
		:add(nn.Threshold(0, 1e-6,true))
		:add(nn.Dropout(0.5))
		:add(nn.Linear(4096, 4096))
		:add(nn.Threshold(0, 1e-6,true))

		--:add(nn.Dropout(0.5))
		:add(nn.Linear(4096, featsize))
		:add(nn.keepBatch())

	local function ConvInit(name)
		for k,v in pairs(alex:findModules(name)) do
			local n = v.kW*v.kH*v.nOutputPlane
			v.weight:normal(0,math.sqrt(2/n))
			if cudnn.version >= 4000 then
				v.bias = nil
				v.gradBias = nil
			else
				v.bias:zero()
			end
		end
	end
	local function BNInit(name)
		for k,v in pairs(alex:findModules(name)) do
			v.weight:fill(1)
			v.bias:zero()
		end
	end
	ConvInit('cudnn.SpatialConvolution')
	ConvInit('nn.SpatialConvolution')
	BNInit('cudnn.SpatialBatchNormalization')
	BNInit('nn.SpatialBatchNormalization')
	for k,v in pairs(alex:findModules('nn.Linear')) do
		v.bias:zero()
	end
	alex:get(1).gradInput = nil

	return alex
end
