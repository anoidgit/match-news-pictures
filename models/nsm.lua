require "dep.negSample"

function getnsm()
	return nn.ParallelTable()
		:add(nn.Identity())
		:add(nn.negSample())
end

function keeplong(nsm)
	nsm:get(2).noise=nsm:get(2).noise:cudaLong()
end
