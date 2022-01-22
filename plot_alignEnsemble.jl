cd("/home/bart/Documents/replicated_boltzmann/test")

push!(LOAD_PATH, joinpath(pwd(), "../src"))
using Revise
using ReplicatedBoltzmann
cd("/home/bart/Documents/robustalignment")
#cd("/home/meynard/Documents/robustalignment")
# cd("/home/bart/Documents/replicated_boltzmann/test")
push!(LOAD_PATH, joinpath(pwd(), "src"))
using RobustAlignment
using Plots
using JLD: save, load
using Random
using FastaIO: writefasta, readfasta
using PlmDCA
using PdbTool
using Distributed
using StatsBase
using Distributions
using ArDCA
using ExtractMacro: @extract
using DCAUtils
using Base.Threads: @threads, nthreads
using DataStructures
using ArgParse

s = ArgParseSettings()
@add_arg_table s begin
    "arg1"
        help = "idx of the family to try"
        required = true
end
parsed_args = parse_args(ARGS, s)
l= parsed_args["arg1"]


function AverageRankVote33(scoresOnPDB::Vector{Any}; numbOfContacts::Int64=0, alpha = 0.0)
	ranks = DefaultDict{Tuple{Int64,Int64},Vector{Float64}}([])
	for seed in 1:length(scoresOnPDB)
		score = scoresOnPDB[seed]
		for pair in 1:length(score)
			push!(ranks[(score[pair][1],score[pair][2])], pair)
		end
	end
	if numbOfContacts == 0
		numbOfContacts = size(collect(keys(ranks)))[1]
	end
	orderedPairs = []
	bestkeylengths = []
	for k in 1:numbOfContacts
		bestkey = (0,0)
		bestscore = Inf
		bestscoreprecision = 0
		for key in keys(ranks)
			pred = mean(ranks[key]) + alpha*(length(scoresOnPDB)-length(ranks[key]))
			if pred<bestscore
				bestkey = key
				bestscore = pred
				bestscoreprecision = length(ranks[key])
			elseif pred==bestscore
				if length(ranks[key])>bestscoreprecision
					bestkey = key
					bestscore = pred
					bestscoreprecision = length(ranks[key])
				end
			end
		end
		# @show length(ranks[bestkey])
		push!(bestkeylengths, length(ranks[bestkey]))
		push!(orderedPairs, bestkey)
		delete!(ranks,bestkey)
	end
	return orderedPairs, bestkeylengths
end


function copelandHeavyside(x::Float64)
	if x>0.0
		return 1.0
	elseif x==0.0
		return 0.5
	else
		return 0.0
	end
end


function Ensemble_ppv_Copeland(scoresOnPDB::Vector{Any}; numbOfContacts::Int64=0)
	ranks = DefaultDict{Tuple{Int64,Int64},Float64}(0.0)
	pairspos  = DefaultDict{Tuple{Int64,Int64},Int64}(0)
	pos = 1
	for seed in 1:length(scoresOnPDB)
		score = scoresOnPDB[seed]
		for pair in 1:length(score)
			ranks[(score[pair][1],score[pair][2])] +=1.0
			if ranks[(score[pair][1],score[pair][2])] ==1.0
				pairspos[(score[pair][1],score[pair][2])] = pos
				pos +=1
			end
		end
	end
	if numbOfContacts == 0
		numbOfContacts = size(collect(keys(ranks)))[1]
	end
	pairsposInv  = DefaultDict{Int64,Tuple{Int64,Int64}}((0,0))
	for key in keys(pairspos)
		pairsposInv[pairspos[key]] = key
	end
	# pairpos = [k for k in keys(ranks)]
	Nkey = length(keys(ranks))
	CopelandMatrix = zeros(Nkey, Nkey)
	for seed in 1:length(scoresOnPDB)
		score = scoresOnPDB[seed]
		@threads for e1 in 1:length(score) -1
			e1pos = pairspos[(score[e1][1],score[e1][2])]
			for e2 in 1:e1-1
				e2pos = pairspos[(score[e2][1],score[e2][2])]
				CopelandMatrix[e1pos, e2pos] += -1
			end
			for e2 in e1+1:length(score)
				e2pos = pairspos[(score[e2][1],score[e2][2])]
				CopelandMatrix[e1pos, e2pos] += +1
			end
		end
	end
	CopelandMatrix .= copelandHeavyside.(CopelandMatrix)
	score = sum(CopelandMatrix, dims=2)
	ranking = ordinalrank(score; rev=true)
	orderedPairs = []
	for k in 1:numbOfContacts
		push!(orderedPairs, (0,0))
	end
	for k in 1:length(ranking)
		if ranking[k]<=numbOfContacts
			orderedPairs[ranking[k]] = pairsposInv[k]
		end
	end
	return orderedPairs
end

conserved= String.(split("ARNDCEQGHILKMFPSTWYV-",""))
conserved2= String.(split("arndceqghilkmfpstwyv.",""))
gap = [".", "-"]
totcoserved = cat(conserved, conserved2;dims=1)
function cleanData(file)
	data = readfasta(file)
	data2 = []
	count = 0
	for i in 1: length(data)
		annot = data[i][1]
		seq = String.(split(data[i][2], ""))
		error = length(findall(x->!(x in totcoserved),seq))
		if error ==0
			push!(data2, (annot,seq))
		else
			count+=1
		end
		@show (file, count)
		writefasta(file, data2)
		@show (file, count)
	end
end

















# listfolder = ["test", "testgal4", "testube", "testbrca"]
list_family = ["PF00107",
# "PF00041",
"PF00028",
"PF00111",
# "PF00043",
"PF00013",
# "PF00018",
"PF00011",
# "PF00035",
"PF00014",
"PF00017",
# "PF00027",
#"PF00084",
"PF00046",
"PF00081",
"PF00105"]

list_pdb = ["1A71.pdb",
#"1BQU.pdb",
"2O72.pdb",
"1A70.pdb",
# "1GSU.pdb",
"1WVN.pdb",
#"2HDA.pdb",
"2bol.pdb",
# "1O0W.pdb",
"5PTI.pdb",
"1O47.pdb",
# "3FHI.pdb",
#"1ELV.pdb",
"2VI6.pdb",
"3BFR.pdb",
"1GDC.pdb"
]

list_chains = ["A",  #ouB
#"A", #ouB
"A",
"A",
# "A", #ouB
"A",
#"A",
"A", #ouB
# "A", #ouB
"A",
"A",
# "B",
#"A",
"A", #ouBCDEFGH
"A",
"A"]



function HMM_Mapping_Avg(pdbpath, hmms_list, chainname, mindist, contactdist)
	pdb=PdbTool.parsePdb(pdbpath)
	pdbSeq=PdbTool.chainSeq(pdb.chain[chainname])
	sizepdb = (pdb.chain[chainname].length)
	hm = zeros(sizepdb,sizepdb)
	contacts = []
	chainresidue = [i for i in 1:10000 if "$i" in keys(pdb.chain[chainname].residue)]
	for i in 1:sizepdb-mindist
		for j in i+mindist:sizepdb
			i_residue = chainresidue[i]
			j_residue = chainresidue[j]
			dist = PdbTool.residueDist(pdb.chain[chainname].residue["$i_residue"], pdb.chain[chainname].residue["$j_residue"])
			if dist<=contactdist
				push!(contacts, (i,j))
			end
		end
	end
	for i=1:length(contacts)
	    x, y = contacts[i]
	    hm[x,y] = 1.0
	end
	for seed in 1:length(hmms_list)
		pdb=PdbTool.parsePdb(pdbpath)
		hmmpath = hmms_list[seed]#"tmp/hmmjh_$(seed)_$(partclustered)"
		PdbTool.mapChainToHmm(pdb.chain[chainname],hmmpath)
		hmmout = PdbTool.mapSeqToHmm(pdbSeq,hmmpath)
		hmmoutreverse = Dict(value => key for (key, value) in hmmout)
		mappedPos = keys(hmmoutreverse)
		for i in 1:sizepdb-mindist
			if i in mappedPos
				for j in i+mindist:sizepdb
					if j in mappedPos
						hm[j,i] += 0.01
					end
				end
			end
		end
	end
	xs = string.(1:sizepdb)
	ys = string.(sizepdb:-1:1)
	# title = "True contact vs average HMM mapping clust=$partclustered"
	pltcon = heatmap(xs,ys,reverse(hm, dims=1); fillcolor=:dense, aspectratio=1, xmirror = true)
	return pltcon
end

function HMM_Mapping_single(pdbpath, hmm_path, chainname, mindist, contactdist)
	pdb=PdbTool.parsePdb(pdbpath)
	pdbSeq=PdbTool.chainSeq(pdb.chain[chainname])
	sizepdb = (pdb.chain[chainname].length)
	hm = zeros(sizepdb,sizepdb)
	contacts = []
	chainresidue = [i for i in 1:10000 if "$i" in keys(pdb.chain[chainname].residue)]
	for i in 1:sizepdb-mindist
		for j in i+mindist:sizepdb
			i_residue = chainresidue[i]
			j_residue = chainresidue[j]
			dist = PdbTool.residueDist(pdb.chain[chainname].residue["$i_residue"], pdb.chain[chainname].residue["$j_residue"])
			if dist<=contactdist
				push!(contacts, (i,j))
			end
		end
	end
	for i=1:length(contacts)
		x, y = contacts[i]
		hm[x,y] = 1.0
	end
	pdb=PdbTool.parsePdb(pdbpath)
	#hmmpath = hmm_path#"jackhmm.hmm"#"HMM_pfam_$(familyName).hmm"
	PdbTool.mapChainToHmm(pdb.chain[chainname],hmm_path)
	hmmout = PdbTool.mapSeqToHmm(pdbSeq,hmm_path)
	hmmoutreverse = Dict(value => key for (key, value) in hmmout)
	mappedPos = keys(hmmoutreverse)
	for i in 1:sizepdb-mindist
		if i in mappedPos
			for j in i+mindist:sizepdb
				if j in mappedPos
					hm[j,i] += 1.0
				end
			end
		end
	end
	xs = string.(1:sizepdb)
	ys = string.(sizepdb:-1:1)
	# title = "True contact vs Pfam HMM mapping"
	pltcon = heatmap(xs,ys,reverse(hm, dims=1); fillcolor=:dense, aspectratio=1, xmirror = true)
	return pltcon
end

function Plot_ConatactMap(pdbpath, orderedpairs, chainname, mindist, contactdist)
	pdb=PdbTool.parsePdb(pdbpath)
	pdbSeq=PdbTool.chainSeq(pdb.chain[chainname])
	sizepdb = (pdb.chain[chainname].length)
	hm = zeros(sizepdb,sizepdb)
	contacts = []
	chainresidue = [i for i in 1:10000 if "$i" in keys(pdb.chain[chainname].residue)]
	for i in 1:sizepdb-mindist
		for j in i+mindist:sizepdb
			i_residue = chainresidue[i]
			j_residue = chainresidue[j]
			dist = PdbTool.residueDist(pdb.chain[chainname].residue["$i_residue"], pdb.chain[chainname].residue["$j_residue"])
			if dist<=contactdist
				push!(contacts, (i,j))
			end
		end
	end
	for i=1:length(contacts)
		x, y = contacts[i]
		hm[x,y] = 1.0
	end
	chainresidue = [i for i in 1:10000 if "$i" in keys(pdb.chain[chainname].residue)]
	for pa in 1:length(orderedpairs)
		i = orderedpairs[pa][1]
		j = orderedpairs[pa][2]
		i_residue = chainresidue[i]
		j_residue = chainresidue[j]
		dist = PdbTool.residueDist(pdb.chain[chainname].residue["$i_residue"], pdb.chain[chainname].residue["$j_residue"])
		if dist<=8.0
			hm[j,i]=1
		else
			hm[j,i]=0.5
		end
	end
	# partclustered = 0.0
	xs = string.(1:sizepdb)
	ys = string.(sizepdb:-1:1)
	# title = "Contact Ensemble Cluster=$partclustered"
	pltcon = heatmap(xs,ys,reverse(hm, dims=1); fillcolor=:dense, aspectratio=1, xmirror = true)
	return pltcon
end






clusterpartlist = [0.0, 0.5, 0.7, 0.9, 1.0]
ncontacts = 500
NSEED=100

cd("/home/bart/Documents/TestAli/test"*list_family[l])
familyName = list_family[l]
pdbpath = list_pdb[l]
hmmpathPFAM = "HMM_pfam_$(familyName).hmm"
msadir = "$(familyName).faa"
pdb=PdbTool.parsePdb(pdbpath)
PdbTool.mapChainToHmm(pdb.chain[list_chains[l]],hmmpathPFAM)
sizepdb = (pdb.chain[list_chains[l]].length)
ncontacts = max(500, 3*sizepdb)
N =  get_HMMlength(hmmpathPFAM)
mypottsplm = PottsModel(N)
plmo = plmdca_asym(joinpath(pwd(), msadir), theta = :auto)
plm_couplings = deflate_matrix(plmo.Jtensor)
plm_fields = plmo.htensor
mypottsplm.fields .= plm_fields
mypottsplm.couplings .= plm_couplings
score, FN, Jtensor, hplm = ComputeScore(mypottsplm, 1)
scoreOnPDB_PFAM = HMM_map_position(score, pdb.chain[list_chains[l]], "HMM_pfam_$(familyName).hmm")
pfammaping = HMM_Mapping_single(pdbpath, "HMM_pfam_$(familyName).hmm", list_chains[l], 5, 8.0)
title!(pfammaping, "Mapping of $(familyName) PFAM HMM on pdb")
savefig(pfammaping, "Plots/pfamMapping$(familyName).png")
pfamcontatcs = Plot_ConatactMap(pdbpath, scoreOnPDB_PFAM[map(x->x[2]-x[1],scoreOnPDB_PFAM).>=5][1:ncontacts], list_chains[l], 5, 8.0)
title!(pfamcontatcs, "Contacts of $(familyName) PFAM HMM on pdb")
savefig(pfamcontatcs, "Plots/ContactsPFAM$(familyName).png")
for partclustered in clusterpartlist
	path =joinpath(pwd(),"rankingContacts_$(partclustered)_$(familyName)" * ".jld")
	dic  = load(path)
	ranking_list = dic["ranking_list"]
	N_SEED = 100
	pdbpath = list_pdb[l]
	pdb=PdbTool.parsePdb(pdbpath)
	sizepdb = (pdb.chain[list_chains[l]].length)
	ncontacts = max(500, 3*sizepdb)
	ppvEs = []
	plt = plot(title="ppv score for alpha $(partclustered) | $(familyName) ")
	for al in 0:2:20
		orderedpairs, bestkeylengths = AverageRankVote33(ranking_list; numbOfContacts=ncontacts, alpha=al)
		ppvE = PPV_fromPDBPOS(pdb.chain[list_chains[l]], orderedpairs, 8.0)
		minplot = min(500, length(ppvE))
		plot!(plt, ppvE[1:minplot], label="$al")
	end
	savefig(plt, "Plots/AvgRank_CompareAlpha_$(familyName)_$(partclustered).png")
	for lim in [5, 10, 20]
		@show lim
		plt = plot(title="ppv score for minsep= $lim clust$(partclustered) | $(familyName) ")
		scoreOnPDBpfam = deepcopy(scoreOnPDB_PFAM)
		scoreOnPDBpfam= scoreOnPDBpfam[map(x->x[2]-x[1],scoreOnPDBpfam).>=lim]
		ppvPFAM = PPV_fromPDBPOS(pdb.chain[list_chains[l]], scoreOnPDBpfam, 8.0)
		plot!(plt, ppvPFAM[1:500], label="PFAM")
		ranking_list = dic["ranking_list"]
		scoresOnPDB = []
		for sc in 1:length(ranking_list)
			score = ranking_list[sc]
			score = score[map(x->x[2]-x[1],score).>=lim]
			push!(scoresOnPDB, score)
		end
		orderedPairsavg, bestkeylengths = AverageRankVote33(scoresOnPDB; numbOfContacts=ncontacts)
		PPVEavg = PPV_fromPDBPOS(pdb.chain[list_chains[l]], orderedPairsavg, 8.0)
		if lim==5
			avgcontatcs = Plot_ConatactMap(pdbpath, orderedPairsavg, list_chains[l], 5, 8.0)
			title!(avgcontatcs, "Contacts of $(familyName) avgRank $(partclustered)  HMM on pdb")
			savefig(avgcontatcs, "Plots/ContactsAvgRank$(partclustered)_$(familyName).png")
		end
		plot!(plt, PPVEavg[1:500], label="AVGrank")
		orderedpairsCop = Ensemble_ppv_Copeland(scoresOnPDB; numbOfContacts=ncontacts)
		# PPVEcop = PPV_fromPDBPOS(pdb.chain[list_chains[l]], orderedpairsCop, 8.0)
		# plot!(plt, PPVEcop[1:500], label="Copeland")
		savefig(plt, "Plots/CompareLimit$(lim)_$(familyName)_$(partclustered).png")
	end
	hmmlists = []
	for seed in 1:NSEED
		push!(hmmlists, "tmp/hmm_$(seed)_$(partclustered)")
	end
	hmmmapavg =  HMM_Mapping_Avg(pdbpath, hmmlists, list_chains[l], 5, 8.0)
	title!(hmmmapavg, "Mapping of $(familyName) avg clust $(partclustered)  HMM on pdb")
	savefig(hmmmapavg, "Plots/AVGMapping$(familyName)_$(partclustered).png")
end
