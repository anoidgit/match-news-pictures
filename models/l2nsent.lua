require "dep.seqvec"

--require "dep.weightSum"--this was needed by weight seqvec model
--require "dep.reWShape"--this was needed by neural weight seqvec model
--require "dep.wseqvec"
--require "dep.neuralwseqvec"

require "dep.stdalex"
require "dep.alex"
--require "dep.vgg"

function getusenn(featsize)
	local wvec=torch.FloatTensor(nword, featsize*2):normal(0, 1)
	wvec[nword]:zero()
	return nn.ParallelTable()
		:add(buildseqv(wvec,featsize))
		:add(buildpicv(featsize))
end
