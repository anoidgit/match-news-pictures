function buildvgg(featsize)

	local backend = cudnn

	local vgg = nn.Sequential()

	-- building block
	local function ConvBNReLU(nInputPlane, nOutputPlane)
		vgg:add(backend.SpatialConvolution(nInputPlane, nOutputPlane, 3,3, 1,1, 1,1))
		vgg:add(nn.SpatialBatchNormalization(nOutputPlane,1e-3))
		vgg:add(nn.ELU(nil,true))
	return vgg
	end

	-- Will use "ceil" MaxPooling because we want to save as much
	-- space as we can
	local MaxPooling = backend.SpatialMaxPooling

	ConvBNReLU(3,64):add(nn.Dropout(0.3))
	ConvBNReLU(64,64)
	vgg:add(MaxPooling(2,2,2,2):ceil())

	ConvBNReLU(64,128):add(nn.Dropout(0.4))
	ConvBNReLU(128,128)
	vgg:add(MaxPooling(2,2,2,2):ceil())

	ConvBNReLU(128,256):add(nn.Dropout(0.4))
	ConvBNReLU(256,256):add(nn.Dropout(0.4))
	ConvBNReLU(256,256)
	vgg:add(MaxPooling(2,2,2,2):ceil())

	ConvBNReLU(256,512):add(nn.Dropout(0.4))
	ConvBNReLU(512,512):add(nn.Dropout(0.4))
	ConvBNReLU(512,512)
	vgg:add(MaxPooling(2,2,2,2):ceil())

	-- In the last block of convolutions the inputs are smaller than
	-- the kernels and cudnn doesn't handle that, have to use cunn
	backend = nn
	ConvBNReLU(512,512):add(nn.Dropout(0.4))
	ConvBNReLU(512,512):add(nn.Dropout(0.4))
	ConvBNReLU(512,512)
	vgg:add(MaxPooling(2,2,2,2):ceil())
	vgg:add(nn.View(512*7*7))
	vgg:add(nn.Dropout(0.5))
	vgg:add(nn.Linear(512*7*7,featsize))
	vgg:add(nn.keepBatch())

	--[[classifier = nn.Sequential()
	classifier:add(nn.Dropout(0.5))
	classifier:add(nn.Linear(512*2*2,512*2*2))
	classifier:add(nn.BatchNormalization(512*2*2))
	classifier:add(nn.ReLU(true))
	classifier:add(nn.Dropout(0.5))
	classifier:add(nn.Linear(512*2*2,121))
	vgg:add(classifier)]]

	-- initialization from MSR
	local function MSRinit(net)
		local function init(name)
			for k,v in pairs(net:findModules(name)) do
				local n = v.kW*v.kH*v.nOutputPlane
				v.weight:normal(0,math.sqrt(2/n))
				v.bias:zero()
			end
		end
		-- have to do for both backends
		init'cudnn.SpatialConvolution'
		init'nn.SpatialConvolution'
	end

	MSRinit(vgg)
	return vgg:cuda()
end
