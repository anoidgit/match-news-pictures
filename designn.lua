require "nn"
require "dpnn"
require "dep.vecLookupZero"
require "dep.keepBatch"
require "cutorch"
require "cunn"
require "cudnn"

require "models.nsm"

function getnn()
	--return getonn()
	return getnnn()
end

function getonn()
	wvec = nil
	--local lmod = loadObject("modrs/nnmod.asc").module
	local lmod = torch.load("modrs/nnmod.asc").module
	return lmod
end

function getnnn()

	require "models.l2nsent"
	return getusenn(256)

end

function getcrit()
	return nn.CosineEmbeddingCriterion();
end
