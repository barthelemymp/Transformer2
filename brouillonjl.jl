function PPV_fromPDBPOS(chain::PdbTool.Chain, orderedPairs::Vector{Any}, distLimit::Float64)
	pv = 0
	ppv = []
	chainresidue = [i for i in 1:10000 if "$i" in keys(chain.residue)]
	for pair in 1:length(orderedPairs)
		i = orderedPairs[pair][1]
		j = orderedPairs[pair][2]
		i_residue = chainresidue[i]
		j_residue = chainresidue[j]
		dist = PdbTool.residueDist(chain.residue["$i_residue"], chain.residue["$j_residue"])
		if dist<=distLimit
			pv +=1
		end
		push!(ppv, pv/pair)
	end
	return ppv
end

function HMM_map_position(score, chain::PdbTool.Chain, hmmfile::String)
	pdbSeq=PdbTool.chainSeq(chain)
	@show pdbSeq
	hmmout = PdbTool.mapSeqToHmm(pdbSeq,hmmfile)
	alignout  = split(PdbTool.alignSeqToHmm(pdbSeq,hmmfile), "")
	@show join(alignout)
	@show join([(i,hmmout[i]) for i in 1:500 if i in keys(hmmout)])
	scoreOnPDB = []
	for i in 1:length(score)
		# @show (score[i][1], score[i][2])
		if alignout[score[i][1]]!="-" && alignout[score[i][2]]!="-"
			iinseq = hmmout[i]
			jinseq = hmmout2[j]
			i_residue = chainresidue[iinseq]
			j_residue = chainresidue[jinseq]

			push!(scoreOnPDB, (hmmout[score[i][1]], hmmout[score[i][2]], score[i][3]))
		end
	end
	return scoreOnPDB
end


alignout
distMat = ones(101,118) * -1.0
for i in 1:101
	for j in 1:118
		if alignout[i]!="-" && alignout2[j]!="-"
			iinseq = hmmout[i]
			jinseq = hmmout2[j]
			i_residue = chainresidue[iinseq]
			j_residue = chainresidue[jinseq]
			distMat[i,j] = PdbTool.residueDist(ch1.residue["$i_residue"], ch2.residue["$j_residue"])
		end
	end
end
