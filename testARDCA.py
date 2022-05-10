#### Test trasformers etc

using Base.Threads: @threads, nthreads
using ArDCA
using DCAUtils
using ArgParse
using ExtractMacro: @extract
using StatsBase
using NPZ

function computeCrossEntropy(arnet::ArNet, testProt::String, testTrans::String)
	@extract arnet:H J p0 idxperm
	# Z = DCAUtils.read_fasta_alignment(fastafile, 1.0)
	Z_Prot = DCAUtils.read_fasta_alignment(testProt, 1.0)#readfasta(testProt)
	Z_Trans = DCAUtils.read_fasta_alignment(testTrans, 1.0)#readfasta(testTrans)
	@show size(Z_Prot)[2], size(Z_Trans)[2]
	@assert size(Z_Prot)[2] == size(Z_Trans)[2]
	CE_mat = zeros(size(Z_Trans)[1]-1, size(Z_Trans)[2])  #sequence_length * nsequences
	accerror = zeros(size(Z_Prot)[2])
    @threads for m in 1:size(Z_Trans)[2] # For every proteins
		inputProt = Z_Prot[:,m]
		TransProt = Z_Trans[:,m]
		TotalProt = cat(inputProt,TransProt, dims=1)

		totH = Vector{Float64}(undef, 21)
		for site in (size(Z_Prot)[1]+1):(length(TotalProt) - 1) #progress in the position #+1 for the sos
			transpos = site - size(Z_Prot)[1] #+ 1
			Js = J[site]
			h = H[site]
			copy!(totH,h)
			for i in 1:site #on all the site before #@avx
				for a in 1:21
					totH[a] += Js[a,TotalProt[i],i]
				end
			end
			p = ArDCA.softmax(totH)
			pred = argmax(p)
			if pred != TotalProt[site+1]
				accerror[m]+=1
			end
			CE_mat[transpos ,m] = - log(p[TotalProt[site+1]])
		end
	end
	return CE_mat, sum(accerror)/size(Z_Prot)[2]
end

function computeCrossEntropyPairs(arnet::ArNet, testProt::String, testTrans::String)
    @extract arnet:H J p0 idxperm
    # Z = DCAUtils.read_fasta_alignment(fastafile, 1.0)
    Z_Prot = DCAUtils.read_fasta_alignment(testProt, 1.0)#readfasta(testProt)
    Z_Trans = DCAUtils.read_fasta_alignment(testTrans, 1.0)#readfasta(testTrans)
    # @show size(Z_Prot)[2], size(Z_Trans)[2]
    @assert size(Z_Prot)[2] == size(Z_Trans)[2]
    scoreHungarian = zeros((size(Z_Trans)[2],size(Z_Trans)[2]))
    CE_mat = zeros(size(Z_Trans)[1]-1, size(Z_Trans)[2])   #sequence_length * nsequences
    for m in 1:size(Z_Trans)[2] # For every proteins
        # @show m
        inputProt = Z_Prot[:,m]
        @threads for j in 1:size(Z_Trans)[2]
            TransProt = Z_Trans[:,j]
            TotalProt = cat(inputProt,TransProt, dims=1)
            totH = Vector{Float64}(undef, 21)
            for site in size(Z_Prot)[1]:(length(TotalProt) - 1) #progress in the position
                transpos = site - size(Z_Prot)[1] + 1
                Js = J[site]
                h = H[site]
                copy!(totH,h)
                for i in 1:site #on all the site before #@avx
                    for a in 1:21
                        totH[a] += Js[a,TotalProt[i],i]
                    end
                end
                p = ArDCA.softmax(totH)

                scoreHungarian[j ,m] -= Float64(log(p[TotalProt[site+1]]))
            end
        end
    end
    return scoreHungarian
end




pathfastatrain = "train_real"

pathfastaval = "val_real"


arnet,arvar=ardca(pathfastatrain*"joined.faa", verbose=false, lambdaJ=0.0001,lambdaH=0.0001; permorder=:NATURAL)

scoreHungarian = computeCrossEntropyPairs(arnet, pathfastaval*"1.faa", pathfastaval*"2.faa")

