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

function parse_commandline()
    s = ArgParseSettings()
    @add_arg_table! s begin
	"pathfastatrain"
        help = "MSA path train"
		arg_type = String
        required = true

	"pathfastatest"
        help = "PDB path"
		arg_type = String
        required = true

	"pathfastaval"
        help = "PDB chain ID"
		arg_type = String
        required = true

	"pathscoreH"
        help = "score Hungarian"
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
pathfastatest = parsed_args["pathfastatest"]
pathfastaval = parsed_args["pathfastaval"]
pathscoreH = parsed_args["pathscoreH"]



# pathfastatrain = "/home/Datasets/DomainsInter/processed/combined_MSA_ddi_$(k)_joined_train_"
# pathfastatest = "/home/Datasets/DomainsInter/processed/combined_MSA_ddi_$(k)_joined_test_"
# pathfastaval = "/home/Datasets/DomainsInter/processed/combined_MSA_ddi_$(k)_joined_val_"
#learn Profile
# alignmentcc = Alignment(pathfastatrain*"2.faa", SymbolMap("ACDEFGHIKLMNPQRSTVWY-"))
# WeightsAlignment!(alignmentcc, :auto)
# profile = ProfileModel(alignmentcc; pc = 0.001)
# #eval Profile
# ceprofileTrain = mean(computeCrossEntropy(profile, pathfastatrain*"2.faa"))
# ceprofileTest = mean(computeCrossEntropy(profile, pathfastatest*"2.faa"))
# ceprofileVal = mean(computeCrossEntropy(profile, pathfastaval*"2.faa"))
# fastafile = "/home/Datasets/DomainsInter/processed/combined_MSA_ddi_$(k)_joined.fasta"
#arnet,arvar=ardca(pathfastatrain*"joined.faa", verbose=false, lambdaJ=0.02,lambdaH=0.001; permorder=:NATURAL)
arnet,arvar=ardca(pathfastatrain*"joined.faa", verbose=false, lambdaJ=0.0001,lambdaH=0.0001; permorder=:NATURAL)

CE_ar_train, acctrain = computeCrossEntropy(arnet, pathfastatrain*"1.faa", pathfastatrain*"2.faa")

CE_ar_test, acctest = computeCrossEntropy(arnet, pathfastatest*"1.faa", pathfastatest*"2.faa")

CE_ar_val, accval = computeCrossEntropy(arnet, pathfastaval*"1.faa", pathfastaval*"2.faa")

scoreHungarian = computeCrossEntropyPairs(arnet, pathfastaval*"1.faa", pathfastaval*"2.faa")
npzwrite(pathscoreH, scoreHungarian)
# temp2train = tempname()
# tempjoinedtrain = tempname()
# temp2test = tempname()
# tempjoinedtest = tempname()
# temp2val = tempname()
# tempjoinedval = tempname()
# randomPairs(pathfastatrain*"1.faa",pathfastatrain*"2.faa", temp2train, tempjoinedtrain)
# randomPairs(pathfastatest*"1.faa",pathfastatest*"2.faa", temp2test, tempjoinedtest)
# randomPairs(pathfastaval*"1.faa",pathfastaval*"2.faa", temp2val, tempjoinedval)

# arnetr,arvarr=ardca(tempjoinedtrain, verbose=false, lambdaJ=0.02,lambdaH=0.001; permorder=:NATURAL)
# CE_ar_train_R = computeCrossEntropy(arnetr, pathfastatrain*"1.faa", temp2train)
# CE_ar_test_R = computeCrossEntropy(arnetr, pathfastatest*"1.faa", temp2test)
# CE_ar_val_R = computeCrossEntropy(arnetr, pathfastaval*"1.faa", temp2val)

# @show (ceprofileTrain, ceprofileTest, ceprofileVal)
@show (mean(CE_ar_train), mean(CE_ar_test), mean(CE_ar_val))
@show (acctrain, acctest, accval)
# @show (mean(CE_ar_train_R), mean(CE_ar_test_R), mean(CE_ar_val_R))
