using Base.Threads: @threads, nthreads
using ArDCA
using DCAUtils
using ArgParse
using ExtractMacro: @extract
using StatsBase
using NPZ
using PdbTool
using PlmDCA


function get_HMMlength(HmmPath)
	a = String(read(`grep LENG $HmmPath`))
	b = split(a, "  ")[2]
	c = split(b, "\n")[1]
	return parse(Int64,c)
end

function makeInterRoc(score::Array{Tuple{Int64,Int64,Float64},1},chain1::PdbTool.Chain,chain2::PdbTool.Chain;sz=200,cutoff::Float64=8.0,out::AbstractString="return",pymolMode::Bool=false,naccessRatio::Float64=1.0)

		# Check if mapping is existent
		if chain1.mappedTo == ""
				error("chain 1 has no mapping")
		elseif chain2.mappedTo == ""
				error("chain 2 has no mapping")
		end
		LENG1=PdbTool.getHmmLength(chain1.mappedTo)
		LENG2=PdbTool.getHmmLength(chain2.mappedTo)

		# Get naccess cutoff if necessary
		if naccessRatio<1.0
			naList1=zeros(LENG1);
			naList2=zeros(LENG2);
			i=1;
			for r1 in values(chain1.align)
				naList1[i]=r1.naccess;
				i+=1;
			end
			i=1;
			for r2 in values(chain2.align)
				naList2[i]=r2.naccess;
				i+=1;
			end
			naList1=sort(naList1,rev=true);
			na1Cutoff=naList1[parse(Int64,round(LENG1*naccessRatio))];
			println(na1Cutoff)
			naList2=sort(naList2,rev=true);
			na2Cutoff=naList2[parse(Int64,round(LENG2*naccessRatio))];
			println(na2Cutoff)
		end

		if out=="return"
			roc=Array{Tuple{AbstractString,AbstractString,Float64,Float64}}(undef,0)
			s::Int64=0
			i::Int64=0
			hits::Int64=0
			positives::Int64=0
			while s<sz && i<size(score,1)
				i+=1
				if score[i][1] <= LENG1 && score[i][2] > LENG1
					ind1=score[i][1]
					ind2=score[i][2]-LENG1
				elseif score[i][2] <= LENG1 && score[i][1] > LENG1
					ind1=score[i][2]
					ind2=score[i][1]-LENG1
				else
					continue
				end
				if haskey(chain1.align,ind1) && haskey(chain2.align,ind2)
					if naccessRatio<1.0
						if chain1.align[ind1].naccess<na1Cutoff || chain2.align[ind2].naccess<na2Cutoff
							continue;
						end
					end
					s+=1
					id1=chain1.align[ind1].identifier
					id2=chain2.align[ind2].identifier
					if PdbTool.residueDist(chain1.align[ind1],chain2.align[ind2])<cutoff
						hits+=1
						if pymolMode
							hits=s
						end
						push!(roc,(id1,id2,hits/s,score[i][3]))
					else
						if pymolMode
							hits=0
						end
						push!(roc,(id1,id2,hits/s,score[i][3]))
					end
				end
			end
			return roc
		end
	end


function mergedPDBInterRoc(score::Array{Tuple{Int64,Int64,Float64},1},chain1_list,chain2_list;sz=200,cutoff::Float64=8.0,out::AbstractString="return",pymolMode::Bool=false,naccessRatio::Float64=1.0)
	# Check if mapping is existent
	for p in 1:length(chain1_list)
		chain1 = chain1_list[p]
		chain2 = chain2_list[p]
		if chain1.mappedTo == ""
				error("chain 1 has no mapping")
		elseif chain2.mappedTo == ""
				error("chain 2 has no mapping")
		end
	end
	LENG1=PdbTool.getHmmLength(chain1_list[1].mappedTo)
	LENG2=PdbTool.getHmmLength(chain2_list[1].mappedTo)
	roc=Array{Tuple{Float64,Float64}}(undef,0)
	s::Int64=0
	i::Int64=0
	hits::Int64=0
	positives::Int64=0
	while s<sz && i<size(score,1)
		i+=1
		tempContact::Int64 = 0
		tempHaskey::Int64 = 0
		if score[i][1] <= LENG1 && score[i][2] > LENG1
			ind1=score[i][1]
			ind2=score[i][2]-LENG1
		elseif score[i][2] <= LENG1 && score[i][1] > LENG1
			ind1=score[i][2]
			ind2=score[i][1]-LENG1
		else
			continue
		end
		for p in 1:length(chain1_list)
			chain1 = chain1_list[p]
			chain2 = chain2_list[p]
			if haskey(chain1.align,ind1) && haskey(chain2.align,ind2)
				tempHaskey +=1
				# s+=1
				id1=chain1.align[ind1].identifier
				id2=chain2.align[ind2].identifier
				if PdbTool.residueDist(chain1.align[ind1],chain2.align[ind2])<cutoff
					tempContact+=1
				end
			end
		end
		if tempHaskey>0
			s+=1
			if tempContact>=1
				hits+=1
				push!(roc,(hits/s,score[i][3]))
			else
				push!(roc,(hits/s,score[i][3]))
			end
		end
	end
	return roc
end


function ppv_mergedcontact(file::String, score, sequence_length1::Int, sequence_length2::Int;contact_dist::Float64=10.0, sizeppv::Int=200)
    A = DelimitedFiles.readdlm(file, ',')
    l, _ =size(A)
    contacts = []#Array{Int, 2}
    distmat = Inf.*ones(sequence_length1,sequence_length2)
    for i=2:l
		@show i, A[i,:]
        indi = A[i,1]+1
        indj =A[i,2]+1
		if A[i,4] != ""
			@show A[i,4]
			@show indi, indj
        	distmat[indi,indj] = min(distmat[indi,indj], A[i,4])
		end
    end
	y = zeros(sizeppv)
	goodpred = 0
    for l=1:sizeppv
		i = score[l][1]
		j = score[l][2]
        if distmat[i,j]<contact_dist
			goodpred +=1
        end
		push!(y,  goodpred/l)
    end
    return y
end


# function ppv(results; size::Int=200)
# 	y = zeros(size)
# 	goodPred=0
# 	for l in 1:size
# 		goodPred+=results[l][3]
# 		y[l] = goodPred/l
# 	end
# 	return y
# end










"/home/feinauer/Datasets/DomainsInter/filters/new_training_set_ddi_num_17_PF03171_PF14226.dat"




function parse_commandline()
    s = ArgParseSettings()
    @add_arg_table! s begin
	"pathfastatrain"
        help = "MSA path train"
		arg_type = String
        required = true

	"pathPDB"
        help = "PDB_list"
		arg_type = String
		required = true

	"chainIN"
        help = "chain for imput list"
		arg_type = String
		required = true

	"chainOUT"
	    help = "chain output list"
		arg_type = String
	    required = true

	"hmmRadical"
        help = "PDB path"
		arg_type = String
        required = true

	"outputpath"
        help = "path for plot output"
		arg_type = String
        required = true

	"mode"
        help = "mode of dca: inter or intra"
		arg_type = String
        required = true

    end
    return parse_args(s)
end

parsed_args = parse_commandline()
println("Parsed args:")
for (arg,val) in parsed_args
    println("  $arg  =>  $val")
end

pathfastatrain = parsed_args["pathfastatrain"]
mode = parsed_args["mode"]

using PyCall
np = pyimport("numpy")

PDBs = np.load("pdblisttemp.npy", allow_pickle=true)
chainIN = np.load("chain1listtemp.npy", allow_pickle=true)
chainOUT = np.load("chain2listtemp.npy", allow_pickle=true)

hmmRadical = parsed_args["hmmRadical"]
hmm1 = hmmRadical * "1.hmm"
hmm2 = hmmRadical * "2.hmm"
hmmjoined = hmmRadical * "joined.hmm"
outputpath = parsed_args["outputpath"]







if mode == "inter"
	write("temp2.fasta",read(`./Unalign  $pathfastatrain`))
	write("temp1.fasta",read(`hmmalign --outformat a2m $hmmjoined temp2.fasta`))
	write("temp2.fasta",read(`./removeInserts temp1.fasta`))

	plmo = plmdca_asym(joinpath(pwd(), "temp2.fasta"), theta = :auto)
	chain1_list = []
	chain2_list = []
	for i in 1:length(PDBs)
		comm = "https://files.rcsb.org/download/" * PDBs[i] * ".pdb"
		if  !isfile(PDBs[i]*".pdb")
			run(`wget $comm`)
		end
		pdb1=PdbTool.parsePdb(PDBs[i]*".pdb")
		pdb2=PdbTool.parsePdb(PDBs[i]*".pdb")
		PdbTool.mapChainToHmm(pdb1.chain[chainIN[i]], hmm1)
		PdbTool.mapChainToHmm(pdb2.chain[chainOUT[i]], hmm2)
		push!(chain1_list, pdb1.chain[chainIN[i]])
		push!(chain2_list, pdb2.chain[chainOUT[i]])
	end
	result = mergedPDBInterRoc( plmo.score,chain1_list, chain2_list;cutoff=10.0, sz=200)
	ppv = map(x->x[1],result)
	np.save(outputpath, ppv)
	# else
	# 	sequence_length1 =  get_HMMlength(hmm1)
	# 	sequence_length2 =  get_HMMlength(hmm2)
	# 	y = ppv_mergedcontact(pathPDB, plmo.score, sequence_length1, sequence_length2;contact_dist=8.0, sizeppv=200)

elseif mode == "intra"
	write("temp2.fasta",read(`./Unalign  $pathfastatrain`))
	write("temp1.fasta",read(`hmmalign --outformat a2m $hmm2 temp2.fasta`))
	write("temp2.fasta",read(`./removeInserts temp1.fasta`))
	pdb=PdbTool.parsePdb(pathPDB)
	PdbTool.mapChainToHmm(pdb.chain[chainOUT], hmm2)
	plmo = plmdca_asym(joinpath(pwd(), "temp2.fasta"), theta = :auto)
	result = makeIntraRoc(plmo.score,pdb.chain[chainOUT])
end

#
#
#
# pathfastatrainjoined = "/home/feinauer/Datasets/DomainsInter/processed/combined_MSA_ddi_972_joined_train_joined.faa"
# pathfastatrain1 = "/home/feinauer/Datasets/DomainsInter/processed/combined_MSA_ddi_972_joined_train_1.faa"
# pathfastatrain2 = "/home/feinauer/Datasets/DomainsInter/processed/combined_MSA_ddi_972_joined_train_2.faa"
#
# pathfastatrainjoined = "/home/feinauer/Datasets/DomainsInter/processed/combined_MSA_ddi_972_joined_trainjoined.faa"
# pathfastatrain1 = "/home/feinauer/Datasets/DomainsInter/processed/combined_MSA_ddi_972_joined_train1.faa"
# pathfastatrain2 = "/home/feinauer/Datasets/DomainsInter/processed/combined_MSA_ddi_972_joined_train2.faa"
#
# run(`hmmbuild --symfrac 0.0 hmm_972_1.hmm $pathfastatrain1`)
# run(`hmmbuild --symfrac 0.0 hmm_972_2.hmm $pathfastatrain2`)
# run(`hmmbuild --symfrac 0.0 hmm_972_joined.hmm $pathfastatrainjoined`)
#
# run(`wget https://files.rcsb.org/download/2VBP.pdb`)
# pathPDB = "1BNC.pdb"
# chainIN = "A"
# chainOUT = "A"
# pdb=PdbTool.parsePdb(pathPDB)
# PdbTool.mapChainToHmm(pdb.chain[chainOUT], "hmm_972_2.hmm")
# pdb.chain[chainOUT]
# plmo = plmdca_asym(joinpath(pwd(), pathfastatrain2), theta = :auto)
#
#
# sequence_length1 =  get_HMMlength("hmm_972_1.hmm")
# sequence_length2 =  get_HMMlength("hmm_972_2.hmm")
# ppv_mergedcontact(pathcontacts,plmo. score, sequence_length1, sequence_length2;contact_dist=0.0, sizeppv=200)
# result = PdbTool.makeIntraRoc(plmo.score,pdb.chain[chainOUT])
#
#
# plmo = plmdca_asym(joinpath(pwd(), "temp2.fasta"), theta = :auto)
# result = PdbTool.makeIntraRoc(plmo.score,pdb.chain[chainOUT])
#
# pathsample = "sample_972_2.faa"
# write("temp2.fasta",read(`./Unalign  $pathsample`))
# write("temp1.fasta",read(`hmmalign --outformat a2m hmm_972_2.hmm temp2.fasta`))
# write("temp2.fasta",read(`./removeInserts temp1.fasta`))
# plmo = plmdca_asym(joinpath(pwd(), "temp2.fasta"), theta = :auto)
# result = PdbTool.makeIntraRoc(plmo.score,pdb.chain[chainOUT])
#
#
#
#
#
#
#
#
# pathfastatrainjoined
# pathPDB = "2VBP.pdb"
# chainIN = "A"
# chainOUT = "A"
# pdb=PdbTool.parsePdb(pathPDB)
# pdb2=PdbTool.parsePdb(pathPDB)
# PdbTool.mapChainToHmm(pdb.chain[chainIN], "hmm_972_1.hmm")
# PdbTool.mapChainToHmm(pdb2.chain[chainOUT], "hmm_972_2.hmm")
# plmo = plmdca_asym(joinpath(pwd(), pathfastatrainjoined), theta = :auto)
# result = makeInterRoc(plmo.score,pdb.chain[chainIN],pdb2.chain[chainOUT])
#
#
#
# pathsample = "sample_972_joined.faa"
#
# plmo = plmdca_asym(joinpath(pwd(),pathsample), theta = :auto)
# result = makeInterRoc(plmo.score,pdb.chain[chainIN],pdb2.chain[chainOUT])
#
#
# pathsample = "sample_972_joined.faa"
# write("temp2.fasta",read(`./Unalign  $pathsample`))
# write("temp1.fasta",read(`hmmalign --outformat a2m hmm_972_joined.hmm temp2.fasta`))
# write("temp2.fasta",read(`./removeInserts temp1.fasta`))
# plmo = plmdca_asym(joinpath(pwd(), "temp2.fasta"), theta = :auto)
# pdb2=PdbTool.parsePdb(pathPDB)
# result = mergedPDBInterRoc( plmo.score,chain1_list, chain2_list;cutoff=12.0, sz=200)
#
#
# write("temp2.fasta",read(`./Unalign  $pathfastatrainjoined`))
# write("temp1.fasta",read(`hmmalign --outformat a2m hmm_972_joined.hmm temp2.fasta`))
# write("temp2.fasta",read(`./removeInserts temp1.fasta`))
# plmo = plmdca_asym(joinpath(pwd(), "temp2.fasta"), theta = :auto)
#
#
# pathcontacts = "/home/feinauer/Datasets/DomainsInter/new_training_set_ddi_num_972_PF02785_PF00289.dat"
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
# PDBs = NPZ.load("pdblisttemp.npy")
#
# chainIN = NPZ.load("chain1listtemp.npy")
# chainOUT = NPZ.load("chain1listtemp.npy")
#
#
#
#
# using PyCall
#
# np = pyimport("numpy")
#
# PDBs = np.load("pdblisttemp.npy", allow_pickle=true)
#
# chainIN = np.load("chain1listtemp.npy", allow_pickle=true)
#
# chainOUT = np.load("chain2listtemp.npy", allow_pickle=true)
#
#
# pathfastatrainjoined = "/home/feinauer/Datasets/DomainsInter/processed/combined_MSA_ddi_972_joined_trainjoined.faa"
# pathfastatrain1 = "/home/feinauer/Datasets/DomainsInter/processed/combined_MSA_ddi_972_joined_train1.faa"
# pathfastatrain2 = "/home/feinauer/Datasets/DomainsInter/processed/combined_MSA_ddi_972_joined_train2.faa"
# pathsample = "sample_972_joined.faa"
#
# plmo = plmdca_asym(joinpath(pwd(), "temp2.fasta"), theta = :auto)
# chain1_list = []
# chain2_list = []
# hmm1 = "hmm_972_1.hmm"
# hmm2 = "hmm_972_2.hmm"
# for i in 1:length(PDBs)
# 	comm = "https://files.rcsb.org/download/" * PDBs[i] * ".pdb"
# 	run(`wget $comm`)
# 	pdb=PdbTool.parsePdb(PDBs[i]*".pdb")
# 	pdb2=PdbTool.parsePdb(PDBs[i]*".pdb")
# 	PdbTool.mapChainToHmm(pdb.chain[chainIN[i]], hmm1)
# 	PdbTool.mapChainToHmm(pdb2.chain[chainOUT[i]], hmm2)
# 	push!(chain1_list, pdb.chain[chainIN[i]])
# 	push!(chain2_list, pdb2.chain[chainOUT[i]])
# end
