#### Test trasformers etc


using Base.Threads: @threads, nthreads
using ArDCA
using DCAUtils
using ArgParse
using ExtractMacro: @extract
using StatsBase
using NPZ

function parse_commandline()
    s = ArgParseSettings()
    @add_arg_table! s begin
	"pathfastatrain"
        help = "MSA path train"
		arg_type = String
        required = true
        
    "pathfastaval"
        help = "PDB chain ID"
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
pathfastaval = parsed_args["pathfastaval"]

timetrain = @elapsed begin
    arnet,arvar=ardca(pathfastatrain*"joined.faa", verbose=false, lambdaJ=0.02,lambdaH=0.001; permorder=:NATURAL)
end

timeinfer = @elapsed begin
    CE_ar_val, accval = computeCrossEntropy(arnet, pathfastaval*"1.faa", pathfastaval*"2.faa")
end 


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
# @show (mean(CE_ar_train), mean(CE_ar_test), mean(CE_ar_val))
# @show (acctrain, acctest, accval)
# @show (mean(CE_ar_train_R), mean(CE_ar_test_R), mean(CE_ar_val_R))
@show(timetrain, timeinfer)
