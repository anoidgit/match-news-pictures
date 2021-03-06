print("set default tensor type to float")
torch.setdefaulttensortype('torch.FloatTensor')

function checkgpu(limit)
	local fmem, totalmem = cutorch.getMemoryUsage()
	local amem = fmem/totalmem
	if amem < limit then
		collectgarbage()
	end
	return amem
end

function feval()
	return _inner_err, _inner_gradParams
end

function gradUpdate(mlpin, x, py, ny, criterionin, lr, optm, bsize, nsm, limit)

	_inner_gradParams:zero()

	local pred=mlpin:forward(x)
	_inner_err=criterionin:forward(pred, py)
	sumErr=sumErr+_inner_err
	local gradCriterion=criterionin:backward(pred, py)
	if bsize>1 then
		local negsample=nsm:forward(pred)
		local _negloss=criterionin:forward(negsample, ny)
		_inner_err=_inner_err+_negloss
		local _ngradCriterion=criterionin:backward(negsample, ny)
		local _gradneg=nsm:backward(pred,_ngradCriterion)
		gradCriterion[1]:add(_gradneg[1])
		gradCriterion[2]:add(_gradneg[2])
	end
	pred=nil
	mlpin:backward(x, gradCriterion)

	checkgpu(limit)
	optm(feval, _inner_params, {learningRate = lr})

	--mlpin:maxParamNorm(2)

end

function evaDev(mlpin, criterionin)
	mlpin:evaluate()
	local serr=0
	local pt=torch.CudaTensor(1):fill(1)
	local dsize=1
	xlua.progress(0, ndev)
	for i=1,ndev do
		local ind = tostring(i)
		local it = devt:read(ind):all():cudaLong()
		local ip = devp:read(ind):all():cuda()
		local bsize = ip:size(1)
		if bsize~=dsize then
			pt:resize(bsize):fill(1)
			dsize=bsize
		end
		serr=serr+criterionin:forward(mlpin:forward({it, ip}), pt)
		xlua.progress(i, ndev)
	end
	mlpin:training()
	return serr/ndev
end

--[[function inirand(cyc)
	cyc=cyc or 8
	for i=1,cyc do
		local sdata=math.random(nsam)
	end
end]]

function saveObject(fname,objWrt)
	torch.save(fname,objWrt)
	--[[local file=torch.DiskFile(fname,'w')
	file:writeObject(tmpod)
	file:close()]]
end

print("pre load package")
require "dpnn"
--require "dp"

print("load settings")
require"aconf"

require "utils.Logger"
logger = Logger(logd.."/"..runid..".log", nil, nil, "w")

logger:log("load data")
require "dloader"

sumErr=0
crithis={}
cridev={}

function train()

	local erate=0
	local edevrate=0
	local storemini=1
	local storedevmini=1
	local minerrate=starterate
	local mindeverrate=minerrate

	logger:log("prepare environment")
	require "paths"
	local savedir="modrs/"..runid.."/"
	paths.mkdir(savedir)

	local memlimit = recyclemem or 0.05

	logger:log("load optim")

	require "getoptim"
	local optmethod=getoptim()

	logger:log("design neural networks and criterion")

	require "designn"
	local nnmod=getnn()

	logger:log(nnmod)
	nnmod:training()

	local critmod=getcrit()

	local nsm=getnsm()

	nnmod:cuda()
	critmod:cuda()
	nsm:cuda()
	keeplong(nsm)

	_inner_params, _inner_gradParams=nnmod:getParameters()
	local savennmod=nn.Serial(nnmod):mediumSerial()

	logger:log("init train")
	local epochs=1
	local lr=modlr

	mindeverrate=evaDev(nnmod,critmod)
	logger:log("Init model Dev:"..mindeverrate)
	mindeverrate=math.huge--this line was wired, to force forget the init state, added by ano

	local pt = torch.CudaTensor(1):fill(1)
	local nt = torch.CudaTensor(1):fill(-1)
	local dsize = 1

	local eaddtrain=ntrain*ieps

	collectgarbage()

	logger:log("start pre train")
	for tmpi=1,warmcycle do
		for tmpj=1,ieps do
			xlua.progress(0, ntrain)
			for i=1,ntrain do
				local ind = tostring(i)
				local it = traint:read(ind):all():cudaLong()
				local ip = trainp:read(ind):all():cuda()
				local bsize = ip:size(1)
				if bsize~=dsize then
					pt:resize(bsize):fill(1)
					nt:resize(bsize):fill(-1)
					dsize=bsize
				end
				gradUpdate(nnmod,{it,ip},pt,nt,critmod,lr,optmethod,bsize,nsm,memlimit)
				xlua.progress(i, ntrain)
			end
		end
		local erate=sumErr/eaddtrain
		if erate<minerrate then
			minerrate=erate
		end
		table.insert(crithis,erate)
		logger:log("epoch:"..tostring(epochs)..",lr:"..lr..",Tra:"..erate)
		sumErr=0
		epochs=epochs+1
	end

	if warmcycle>0 then
		logger:log("save neural network trained")
		--savennmod:clearState()
		saveObject(savedir.."nnmod.asc",savennmod)
	end

	epochs=1
	local icycle=1

	local aminerr=1
	local amindeverr=1
	local lrdecayepochs=1

	local cntrun=true

	collectgarbage()

	while cntrun do
		logger:log("start innercycle:"..icycle)
		for innercycle=1,gtraincycle do
			for tmpi=1,ieps do
				xlua.progress(0, ntrain)
				for i=1,ntrain do
					local ind = tostring(i)
					local it = traint:read(ind):all():cudaLong()
					local ip = trainp:read(ind):all():cuda()
					local bsize = ip:size(1)
					if bsize~=dsize then
						pt:resize(bsize):fill(1)
						nt:resize(bsize):fill(-1)
						dsize=bsize
					end
					gradUpdate(nnmod,{it,ip},pt,nt,critmod,lr,optmethod,bsize,nsm,memlimit)
					xlua.progress(i, ntrain)
				end
			end
			local erate=sumErr/eaddtrain
			table.insert(crithis,erate)
			local edevrate=evaDev(nnmod,critmod)
			table.insert(cridev,edevrate)
			logger:log("epoch:"..tostring(epochs)..",lr:"..lr..",Tra:"..erate..",Dev:"..edevrate)
			--logger:log("epoch:"..tostring(epochs)..",lr:"..lr..",Tra:"..erate)
			local modsavd=false
			if edevrate<mindeverrate then
				mindeverrate=edevrate
				amindeverr=1
				aminerr=1--reset aminerr at the same time
				--savennmod:clearState()
				saveObject(savedir.."devnnmod"..storedevmini..".asc",savennmod)
				storedevmini=storedevmini+1
				if storedevmini>csave then
					storedevmini=1
				end
				modsavd=true
				logger:log("new minimal dev error found, model saved")
			else
				if earlystop and amindeverr>earlystop then
					logger:log("early stop")
					cntrun=false
					break
				end
				amindeverr=amindeverr+1
			end
			if erate<minerrate then
				minerrate=erate
				aminerr=1
				if not modsavd then
					--savennmod:clearState()
					saveObject(savedir.."nnmod"..storemini..".asc",savennmod)
					storemini=storemini+1
					if storemini>csave then
						storemini=1
					end
					logger:log("new minimal error found, model saved")
				end
			else
				if aminerr>=expdecaycycle then
					aminerr=0
					if lrdecayepochs>lrdecaycycle then
						modlr=lr
						lrdecayepochs=1
					end
					lrdecayepochs=lrdecayepochs+1
					lr=modlr/(lrdecayepochs)
				end
				aminerr=aminerr+1
			end
			sumErr=0
			epochs=epochs+1
		end

		logger:log("save neural network trained")
		--savennmod:clearState()
		saveObject(savedir.."nnmod.asc",savennmod)

		logger:log("save criterion history trained")
		local critensor=torch.Tensor(crithis)
		saveObject(savedir.."crit.asc",critensor)
		local critdev=torch.Tensor(cridev)
		saveObject(savedir.."critdev.asc",critdev)

		--[[logger:log("plot and save criterion")
		gnuplot.plot(critensor)
		gnuplot.figprint(savedir.."crit.png")
		gnuplot.figprint(savedir.."crit.eps")
		gnuplot.plotflush()
		gnuplot.plot(critdev)
		gnuplot.figprint(savedir.."critdev.png")
		gnuplot.figprint(savedir.."critdev.eps")
		gnuplot.plotflush()]]

		critensor=nil
		critdev=nil

		logger:log("task finished!Minimal error rate:"..minerrate.."	"..mindeverrate)
		--logger:log("task finished!Minimal error rate:"..minerrate)

		logger:log("wait for test, neural network saved at nnmod*.asc")

		icycle=icycle+1

		logger:log("collect garbage")
		collectgarbage()

	end
end

train()
logger:shutDown()
