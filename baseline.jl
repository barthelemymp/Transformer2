#### Test trasformers etc
cd("C:\\Users\\bartm\\Documents\\These\\replicated_boltzmann\\test")
push!(LOAD_PATH, joinpath(pwd(), "../src"))
using Revise
using ReplicatedBoltzmann
using Plots
using JLD: save, load
using Random
using StatsPlots
using DataFrames
using Statistics
using FastaIO: writefasta, readfasta
using PlmDCA
using StatsBase
using ArDCA
using ExtractMacro: @extract
using DCAUtils
using DelimitedFiles

using Base.Threads: @threads, nthreads
cd("C:\\Users\\bartm\\Documents\\These\\Transformer")

function csv2fasta(file::String)
    data = readdlm(file*".csv", ',')
    M = size(data)[1]
    data1 = []
    data2 = []
    datajoin = []
    annot = "pair_"
    join(split(data[3,1]," "))
    for i in 1:M
        push!(data1, (annot*"$i", join(split(data[i,1]," "))))
        push!(data2, (annot*"$i", join(split(data[i,2]," "))))
        push!(datajoin, (annot*"$i", join(split(data[i,1]," "))*join(split(data[i,2]," "))))
    end
    writefasta(file*"_1.faa", data1)
    writefasta(file*"_2.faa", data2)
    writefasta(file*"_joined.faa", datajoin)
end


function computeCrossEntropy(arnet::ArNet, testProt::String, testTrans::String)
    @extract arnet:H J p0 idxperm
    # Z = DCAUtils.read_fasta_alignment(fastafile, 1.0)
    Z_Prot = DCAUtils.read_fasta_alignment(testProt, 1.0)#readfasta(testProt)
    Z_Trans = DCAUtils.read_fasta_alignment(testTrans, 1.0)#readfasta(testTrans)
    @show size(Z_Prot)[2], size(Z_Trans)[2]
    @assert size(Z_Prot)[2] == size(Z_Trans)[2]
    CE_mat = zeros(size(Z_Trans))  #sequence_length * nsequences
    for m in 1:size(Z_Trans)[2] # For every proteins
        inputProt = Z_Prot[:,m]
        TransProt = Z_Trans[:,m]
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
            CE_mat[transpos ,m] = - log(p[TotalProt[site+1]])
        end
    end
    return CE_mat
end



"A Profile Model"
mutable struct ProfileModel
    sequence_length::Integer
    nsymbols::Integer
    fields::Matrix{T} where T <: Number
end


"Constructor that creates a random Potts Model in zero-q gauge."
function ProfileModel(alignment::Alignment; pc::Float64= 0.001)
    freq = single_frequencies(alignment)
    freq .= (1.0 - pc).*freq .+(pc/alignment.nsymbols)
    fields = log.(freq)
    return ProfileModel(alignment.sequence_length, alignment.nsymbols, fields)
end

function getProba(profile::ProfileModel)
    proba = zeros(size(profile.fields))
    for k in 1:profile.sequence_length
        localfields=profile.fields[:,k]
        expF = exp.(localfields)
        proba[:,k] .= expF./sum(expF)
    end
    return proba
end


function computeCrossEntropy(profile::ProfileModel, testTrans::String)

    Z_Trans = DCAUtils.read_fasta_alignment(testTrans, 1.0)#readfasta(testTrans)
    CE_mat = zeros(size(Z_Trans))  #sequence_length * nsequences
    profileproba = getProba(profile)
    for m in 1:size(Z_Trans)[2] # For every proteins
        TransProt = Z_Trans[:,m]
        for site in 1:profile.sequence_length #progress in the position

            p = profileproba[:,site]
            CE_mat[site ,m] = - log(p[TransProt[site]])
        end
    end
    return CE_mat
end





csv2fasta("train_ddi")
csv2fasta("test_ddi")
csv2fasta("val_ddi")
fastafile = joinpath(pwd(),"train_ddi_joined.faa")
arnet,arvar=ardca(fastafile, verbose=false, lambdaJ=0.02,lambdaH=0.001; permorder=:NATURAL)

testProt = "test_ddi_1.faa"
testTrans = "test_ddi_2.faa"
CEtest = computeCrossEntropy(arnet, testProt, testTrans)
mean(CEtest)

testProt = "val_ddi_1.faa"
testTrans = "val_ddi_2.faa"
CEval = computeCrossEntropy(arnet, testProt, testTrans)
mean(CEval)

### HK RR
csv2fasta("train_real")
csv2fasta("test_real")
csv2fasta("val_real")
csv2fasta("val_shuffled")

fastafile = joinpath(pwd(),"train_real_joined.faa")
arnet,arvar=ardca(fastafile, verbose=false, lambdaJ=0.02,lambdaH=0.001; permorder=:NATURAL)


testProt = "test_real_1.faa"
testTrans = "test_real_2.faa"
CEtest = computeCrossEntropy(arnet, testProt, testTrans)
mean(CEtest)
exp(-mean(CEtest))


testProt = "val_real_1.faa"
testTrans = "val_real_2.faa"
CEval = computeCrossEntropy(arnet, testProt, testTrans)
mean(CEval)
exp(-mean(CEval))

testProt = "val_shuffled_1.faa"
testTrans = "val_shuffled_2.faa"
CEval_shuffled = computeCrossEntropy(arnet, testProt, testTrans)
mean(CEval_shuffled)
exp(-mean(CEval_shuffled))




### PPI
using Random
function createRandom(tranpath::String)
    data = readfasta(tranpath*".faa")
    data2 = []
    for i in 1:length(data)
        annot = data[i][1]
        seq = String.(split(data[i][2], ""))
        shuffle!(seq)
        push!(data2, (annot, join(seq)))
    end
    writefasta(tranpath*"_random.faa", data2)
end

createRandom("train_pp1_joined")


csv2fasta("train_pp1")
csv2fasta("test_pp1")
csv2fasta("val_pp1")

fastafile = joinpath(pwd(),"train_pp1_joined.faa")
arnet,arvar=ardca(fastafile, verbose=false, lambdaJ=0.02,lambdaH=0.001; permorder=:NATURAL)


arnet_R,arvar_R=ardca("train_pp1_joined_random.faa", verbose=false, lambdaJ=0.02,lambdaH=0.001; permorder=:NATURAL)


trainProt = "train_pp1_1.faa"
trainTrans = "train_pp1_2.faa"
CEtest = computeCrossEntropy(arnet_R, trainProt, trainTrans)
mean(CEtest)
exp(-mean(CEtest))



testProt = "test_pp1_1.faa"
testTrans = "test_pp1_2.faa"
CEtest = computeCrossEntropy(arnet, testProt, testTrans)
mean(CEtest)
exp(-mean(CEtest))
alignmentcc = Alignment(trainTrans, SymbolMap("ACDEFGHIKLMNPQRSTVWY-"))
# alignmentTrain = Alignment("train_pp1_joined.faa", SymbolMap("ACDEFGHIKLMNPQRSTVWY-"))
WeightsAlignment!(alignmentcc, :auto)
profile = ProfileModel(alignmentcc; pc = 0.001)
ceprofile = mean(computeCrossEntropy(profile, testTrans))

res = arnet(arvar.Z)
loglike = sum(log.(res),dims=1)

alignmentTest = Alignment("test_pp1_joined.faa", SymbolMap("ACDEFGHIKLMNPQRSTVWY-"))
alignmentTest_2 = Alignment("test_pp1_2.faa", SymbolMap("ACDEFGHIKLMNPQRSTVWY-"))
res = arnet(alignmentTest.sequences)
loglike = sum(log.(res),dims=1)

CEtest = computeCrossEntropy(arnet_R, testProt, testTrans)
mean(CEtest)
exp(-mean(CEtest))






testProt = "val_pp1_1.faa"
testTrans = "val_pp1_2.faa"
CEval = computeCrossEntropy(arnet, testProt, testTrans)
mean(CEval)
exp(-mean(CEval))

testProt = "val_shuffled_1.faa"
testTrans = "val_shuffled_2.faa"
CEval_shuffled = computeCrossEntropy(arnet, testProt, testTrans)
mean(CEval_shuffled)
exp(-mean(CEval_shuffled))



















#########################################################################################
#########################################################################################
#########################################################################################


cd("/home/bart/Documents/replicated_boltzmann/test")
push!(LOAD_PATH, joinpath(pwd(), "../src"))
using Revise
using ReplicatedBoltzmann
using Plots
using JLD: save, load
using Random
using StatsPlots
using DataFrames
using Statistics
using FastaIO: writefasta, readfasta
using PlmDCA
using StatsBase
using ArDCA
using ExtractMacro: @extract
using DCAUtils
using DelimitedFiles

#
# cd("C:\\Users\\bartm\\Documents\\These\\Transformer")

function csv2fasta(file::String)
    data = readdlm(file*".csv", ',')
    M = size(data)[1]
    data1 = []
    data2 = []
    datajoin = []
    annot = "pair_"
    join(split(data[3,1]," "))
    for i in 1:M
        push!(data1, (annot*"$i", join(split(data[i,1]," "))))
        push!(data2, (annot*"$i", join(split(data[i,2]," "))))
        push!(datajoin, (annot*"$i", join(split(data[i,1]," "))*join(split(data[i,2]," "))))
    end
    writefasta(file*"_1.faa", data1)
    writefasta(file*"_2.faa", data2)
    writefasta(file*"_joined.faa", datajoin)
end


function computeCrossEntropy(arnet::ArNet, testProt::String, testTrans::String)
    @extract arnet:H J p0 idxperm
    # Z = DCAUtils.read_fasta_alignment(fastafile, 1.0)
    Z_Prot = DCAUtils.read_fasta_alignment(testProt, 1.0)#readfasta(testProt)
    Z_Trans = DCAUtils.read_fasta_alignment(testTrans, 1.0)#readfasta(testTrans)
    @show size(Z_Prot)[2], size(Z_Trans)[2]
    @assert size(Z_Prot)[2] == size(Z_Trans)[2]
    CE_mat = zeros(size(Z_Trans))  #sequence_length * nsequences
    @threads for m in 1:size(Z_Trans)[2] # For every proteins
        inputProt = Z_Prot[:,m]
        TransProt = Z_Trans[:,m]
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
            CE_mat[transpos ,m] = - log(p[TotalProt[site+1]])
        end
    end
    return CE_mat
end



"A Profile Model"
mutable struct ProfileModel
    sequence_length::Integer
    nsymbols::Integer
    fields::Matrix{T} where T <: Number
end


"Constructor that creates a random Potts Model in zero-q gauge."
function ProfileModel(alignment::Alignment; pc::Float64= 0.001)
    freq = single_frequencies(alignment)
    freq .= (1.0 - pc).*freq .+(pc/alignment.nsymbols)
    fields = log.(freq)
    return ProfileModel(alignment.sequence_length, alignment.nsymbols, fields)
end

function getProba(profile::ProfileModel)
    proba = zeros(size(profile.fields))
    for k in 1:profile.sequence_length
        localfields=profile.fields[:,k]
        expF = exp.(localfields)
        proba[:,k] .= expF./sum(expF)
    end
    return proba
end


function computeCrossEntropy(profile::ProfileModel, testTrans::String)

    Z_Trans = DCAUtils.read_fasta_alignment(testTrans, 1.0)#readfasta(testTrans)
    CE_mat = zeros(size(Z_Trans))  #sequence_length * nsequences
    profileproba = getProba(profile)
    for m in 1:size(Z_Trans)[2] # For every proteins
        TransProt = Z_Trans[:,m]
        for site in 1:profile.sequence_length #progress in the position

            p = profileproba[:,site]
            CE_mat[site ,m] = - log(p[TransProt[site]])
        end
    end
    return CE_mat
end





for k in 1:6230
    path = "/home/Datasets/DomainsInter/processed/combined_MSA_ddi_$(k)_2.fasta"
    if  isfile(path)
        data = readfasta(path)
        len = length(split(data[i][2])
    end
end

 isfile("foo.txt")
score = []

 for k in 1:6230
     path = "/home/Datasets/DomainsInter/processed/combined_MSA_ddi_$(k)_2.fasta"
     if  isfile(path)
         @show k
         alignmentcc = Alignment(path, SymbolMap("ACDEFGHIKLMNPQRSTVWY-"))
         WeightsAlignment!(alignmentcc, :auto)
         profile = ProfileModel(alignmentcc; pc = 0.001)
         ceprofile = mean(computeCrossEntropy(profile, path))

         fastafile = "/home/Datasets/DomainsInter/processed/combined_MSA_ddi_$(k)_joined.fasta"
         arnet,arvar=ardca(fastafile, verbose=false, lambdaJ=0.02,lambdaH=0.001; permorder=:NATURAL)
         testProt=  "/home/Datasets/DomainsInter/processed/combined_MSA_ddi_$(k)_1.fasta"
         testTrans =  "/home/Datasets/DomainsInter/processed/combined_MSA_ddi_$(k)_2.fasta"
         CEtest = computeCrossEntropy(arnet, testProt, testTrans)
         ceAR = mean(CEtest)
         @show(k, ceprofile, ceAR)
         push!(score, (k, ceprofile, ceAR))
     end
 end

path = "/home/Datasets/DomainsInter/raw/combined_MSA_ddi_$k.fasta"
data = readfasta(path)
conserved= String.(split("ARNDCEQGHILKMFPSTWYV-",""))
gap = [".", "-"]

function rmvIns(seq)
    seqout = copy(seq)
    seqout = deleteat!(seqout, findall(x->!(x in conserved),seq))
    return seqout
end
 for i in 1:length(data)
    @show length(rmvIns(split(data[i][2], ""))
end






path = "C:\\Users\\bartm\\Documents\\These\\Transformer\\train_real_joined.faa"

alignmentcc = Alignment(path, SymbolMap("ACDEFGHIKLMNPQRSTVWY-"))
WeightsAlignment!(alignmentcc, :auto)
profile = ProfileModel(alignmentcc; pc = 0.001)
ceprofile = mean(computeCrossEntropy(profile, path))

fastafile = "/home/Datasets/DomainsInter/processed/combined_MSA_ddi_$(k)_joined.fasta"
arnet,arvar=ardca(fastafile, verbose=false, lambdaJ=0.02,lambdaH=0.001; permorder=:NATURAL)
testProt=  "/home/Datasets/DomainsInter/processed/combined_MSA_ddi_$(k)_1.fasta"
testTrans =  "/home/Datasets/DomainsInter/processed/combined_MSA_ddi_$(k)_2.fasta"
CEtest = computeCrossEntropy(arnet, testProt, testTrans)
ceAR = mean(CEtest)
@show(k, ceprofile, ceAR)
push!(score, (k, ceprofile, ceAR))





seqinf = readdlm("/home/Datasets/DomainsInter/PPIraw/length.txt")


for k in 1:32
    @show seqid = seqinf[k,1]
    path = "/home/Datasets/DomainsInter/PPIprocessed/PPI_$(k)_joined.fasta"
    if  isfile(path)
        @show k
        alignmentcc = Alignment(path, SymbolMap("ACDEFGHIKLMNPQRSTVWY-"))
        WeightsAlignment!(alignmentcc, :auto)
        profile = ProfileModel(alignmentcc; pc = 0.001)
        ceprofile = mean(computeCrossEntropy(profile, path))

        fastafile = "/home/Datasets/DomainsInter/PPIprocessed/PPI_$(k)_joined.fasta"
        arnet,arvar=ardca(fastafile, verbose=false, lambdaJ=0.02,lambdaH=0.001; permorder=:NATURAL)
        testProt=  "/home/Datasets/DomainsInter/PPIprocessed/PPI_$(k)_1.fasta"
        testTrans =  "/home/Datasets/DomainsInter/PPIprocessed/PPI_$(k)_2.fasta"
        CEtest = computeCrossEntropy(arnet, testProt, testTrans)
        ceAR = mean(CEtest)
        @show(k, ceprofile, ceAR)
        push!(score, (k, ceprofile, ceAR))
    end
end











######### Clustering
using Clustering

path = "C:\\Users\\bartm\\Documents\\These\\Transformer\\train_real_joined.faa"
alignment_joined = Alignment(path, SymbolMap("ACDEFGHIKLMNPQRSTVWY-"))

path = "C:\\Users\\bartm\\Documents\\These\\Transformer\\train_real_1.faa"
alignment_1 = Alignment(path, SymbolMap("ACDEFGHIKLMNPQRSTVWY-"))

path = "C:\\Users\\bartm\\Documents\\These\\Transformer\\train_real_2.faa"
alignment_2 = Alignment(path, SymbolMap("ACDEFGHIKLMNPQRSTVWY-"))

M = alignment_joined.nsequences

alignment_1.sequence_length


Dist_Matrix_joined = zeros(M,M)
Dist_Matrix_1 = zeros(M,M)
Dist_Matrix_2 = zeros(M,M)

# Cluster_truth = zeros(M)
for i in 1:M
	@show i
	seqi_1 = alignment_1.sequences[:,i]
    seqi_2 = alignment_2.sequences[:,i]
    seqi_joined = alignment_joined.sequences[:,i]
	# Cluster_truth[i] = determinecluster(alrand[i])
	for j in 1:M
        seqj_1 = alignment_1.sequences[:,j]
        seqj_2 = alignment_2.sequences[:,j]
        seqj_joined = alignment_joined.sequences[:,j]
		d = sum(seqi_1.!=seqj_1)
		Dist_Matrix_1[i,j] =d
        d = sum(seqi_2.!=seqj_2)
        Dist_Matrix_2[i,j] =d
        d = sum(seqi_joined.!=seqj_joined)
        Dist_Matrix_joined[i,j] =d
	end
end


hclu_1  = hclust(Dist_Matrix_1; linkage=:average )
Dist_Matrix_1_ordered = zeros(M, M)

for i in 1:size(hclu_1.order)[1]
	@show i
	seqi = alignment_1.sequences[:,hclu_1.order[i]]
	for j in 1:M
		seqj = alignment_1.sequences[:,hclu_1.order[j]]
		Dist_Matrix_1_ordered[i,j] = sum(seqi.!=seqj)  #Dist_Matrix[alrand[i],alrand[j]]
	end
end



hclu_2  = hclust(Dist_Matrix_2; linkage=:average )
Dist_Matrix_2_ordered = zeros(M, M)

for i in 1:size(hclu_2.order)[1]
	@show i
	seqi = alignment_2.sequences[:,hclu_2.order[i]]
	for j in 1:M
		seqj = alignment_2.sequences[:,hclu_2.order[j]]
		Dist_Matrix_2_ordered[i,j] = sum(seqi.!=seqj)  #Dist_Matrix[alrand[i],alrand[j]]
	end
end

md1 = mean(Dist_Matrix_1)
Dist_Matrix_12 = copy(Dist_Matrix_2_ordered)./mean(Dist_Matrix_2)

for i in 1:size(hclu_2.order)[1]
	@show i
	seqi = alignment_1.sequences[:,hclu_2.order[i]]
	for j in i:M
		seqj = alignment_1.sequences[:,hclu_2.order[j]]
		Dist_Matrix_12[i,j] = sum(seqi.!=seqj)/md1 #Dist_Matrix[alrand[i],alrand[j]]
	end
end



hm_1 = heatmap(1:M,
    1:M, Dist_Matrix_1_ordered,
    size = (4500,4000), tickfontsize=35, title = "Hamming distance Clustering Protein 1", titlefontsize=55)
savefig(hm_1, "HammingDistanceHM_Translation_AVGClusteringOrder_1.png")

hm_2 = heatmap(1:M,
    1:M, Dist_Matrix_2_ordered,
    size = (4500,4000), tickfontsize=35, title =  "Hamming distance Clustering Protein 2", titlefontsize=55)
savefig(hm_2, "HammingDistanceHM_Translation_AVGClusteringOrder_2.png")

hm_12 = heatmap(1:M,
    1:M, Dist_Matrix_12,
    size = (4500,4000), tickfontsize=35, title = "Hamming distance Clustering Order of protein 2", titlefontsize=55, xlabel="Protein2", ylabel="protein 1")
savefig(hm_12, "HammingDistanceHM_Translation_AVGClusteringOrder_2ordervs1rescaled.png")
matVecV(vector) = reshape(vector, length(vector), 1)
ClV = heatmap(matVecV(Cluster_truth), aspect_ratio = 0.01, yticks=false, size = ( 200, 4000), colorbar=false, c=cgrad([:green, :darkorange,:black]))
plt = plot(hm_12, layout=grid(1, 1, widths=[0.95], height=[0.95]),  xlabel="Protein2", ylabel="protein 1")#,p22,ClH; layout=(2,2))
savefig(plt, "HammingDistanceHM_Inherent_SingularClusteringOrder.png")
display(plt)




#
# (1, 1.4615260208828669, 0.36277606835216575)
# (2, 1.2668464851261334, 0.23916522523631606)
# (3, 1.6300758540327702, 0.421029210511007)
# (5, 1.5496199708620948, 0.30811823959780027)
# (7, 1.2780503283970739, 0.2453644116351409)
# (8, 1.7989406713007878, 0.3812970711420464)
# (9, 1.7715867697652063, 0.34835437224412547)
# (10, 1.6048670099877986, 0.4132353129461116)
# (12, 1.6086577173856516, 0.394732614727519)
# (16, 1.6080177215802776, 0.8305048412079279)
# (19, 1.4140455040222872, 0.4905095786299233)
# (21, 1.3536770782680265, 0.40079940642411555)
# (22, 2.033577029324878, 1.0412198935620631)
# (27, 1.644151802785219, 0.4231673632044662)
# (31, 1.5929814976197807, 0.43935607250904235)

using Random
using FastaIO: writefasta, readfasta
function randomPairs(path1,path2, temp2, tempjoined)
    data1 = readfasta(path1)
    data2 = readfasta(path2)
    dataMixed2 = []
    dataMixedjoined = []
    rp = randperm(length(data1))
    for i in 1:length(rp)
        a1, seq1 = data1[i]
        a2, seq2 = data2[rp[i]]
        push!(dataMixed2, data2[rp[i]])
        push!(dataMixedjoined, (a1, seq1*seq2))
    end
    writefasta(temp2, dataMixed2)
    writefasta(tempjoined, dataMixedjoined)
end




to_check = [17,46,69,71,157,160,251,258]


for k in to_check
    # pathtrain = "/home/Datasets/DomainsInter/processed/combined_MSA_ddi_$(k)_joined_train"
    # pathtest = "/home/Datasets/DomainsInter/processed/combined_MSA_ddi_$(k)_joined_test"
    # pathval = "/home/Datasets/DomainsInter/processed/combined_MSA_ddi_$(k)_joined_val"
    # csv2fasta(pathtrain)
    # csv2fasta(pathtest)
    # csv2fasta(pathval)
    @show k
    pathfastatrain = "/home/Datasets/DomainsInter/processed/combined_MSA_ddi_$(k)_joined_train_"
    pathfastatest = "/home/Datasets/DomainsInter/processed/combined_MSA_ddi_$(k)_joined_test_"
    pathfastaval = "/home/Datasets/DomainsInter/processed/combined_MSA_ddi_$(k)_joined_val_"
    #learn Profile
    alignmentcc = Alignment(pathfastatrain*"2.faa", SymbolMap("ACDEFGHIKLMNPQRSTVWY-"))
    WeightsAlignment!(alignmentcc, :auto)
    profile = ProfileModel(alignmentcc; pc = 0.001)
    #eval Profile
    ceprofileTrain = mean(computeCrossEntropy(profile, pathfastatrain*"2.faa"))
    ceprofileTest = mean(computeCrossEntropy(profile, pathfastatest*"2.faa"))
    ceprofileVal = mean(computeCrossEntropy(profile, pathfastaval*"2.faa"))
    # fastafile = "/home/Datasets/DomainsInter/processed/combined_MSA_ddi_$(k)_joined.fasta"
    arnet,arvar=ardca(pathfastatrain*"joined.faa", verbose=false, lambdaJ=0.02,lambdaH=0.001; permorder=:NATURAL)
    CE_ar_train = computeCrossEntropy(arnet, pathfastatrain*"1.faa", pathfastatrain*"2.faa")
    CE_ar_test = computeCrossEntropy(arnet, pathfastatest*"1.faa", pathfastatest*"2.faa")
    CE_ar_val = computeCrossEntropy(arnet, pathfastaval*"1.faa", pathfastaval*"2.faa")

    temp2train = tempname()
    tempjoinedtrain = tempname()
    temp2test = tempname()
    tempjoinedtest = tempname()
    temp2val = tempname()
    tempjoinedval = tempname()
    randomPairs(pathfastatrain*"1.faa",pathfastatrain*"2.faa", temp2train, tempjoinedtrain)
    randomPairs(pathfastatest*"1.faa",pathfastatest*"2.faa", temp2test, tempjoinedtest)
    randomPairs(pathfastaval*"1.faa",pathfastaval*"2.faa", temp2val, tempjoinedval)

    arnetr,arvarr=ardca(tempjoinedtrain, verbose=false, lambdaJ=0.02,lambdaH=0.001; permorder=:NATURAL)
    CE_ar_train_R = computeCrossEntropy(arnetr, pathfastatrain*"1.faa", temp2train)
    CE_ar_test_R = computeCrossEntropy(arnetr, pathfastatest*"1.faa", temp2test)
    CE_ar_val_R = computeCrossEntropy(arnetr, pathfastaval*"1.faa", temp2val)

    @show (ceprofileTrain, ceprofileTest, ceprofileVal)
    @show (mean(CE_ar_train), mean(CE_ar_test), mean(CE_ar_val))
    @show (mean(CE_ar_train_R), mean(CE_ar_test_R), mean(CE_ar_val_R))
end





















################  make CEloss

cd("/home/bart/Documents/Transformer")
function computeCrossEntropyPairs(arnet::ArNet, testProt::String, testTrans::String)
    @extract arnet:H J p0 idxperm
    # Z = DCAUtils.read_fasta_alignment(fastafile, 1.0)
    Z_Prot = DCAUtils.read_fasta_alignment(testProt, 1.0)#readfasta(testProt)
    Z_Trans = DCAUtils.read_fasta_alignment(testTrans, 1.0)#readfasta(testTrans)
    @show size(Z_Prot)[2], size(Z_Trans)[2]
    @assert size(Z_Prot)[2] == size(Z_Trans)[2]
    scoreHungarian = zeros((size(Z_Trans)[2],size(Z_Trans)[2]))
    CE_mat = zeros(size(Z_Trans))  #sequence_length * nsequences
    for m in 1:size(Z_Trans)[2] # For every proteins
        @show m
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



# pathtrain = "/home/Datasets/DomainsInter/processed/combined_MSA_ddi_$(k)_joined_train"
# pathtest = "/home/Datasets/DomainsInter/processed/combined_MSA_ddi_$(k)_joined_test"
# pathval = "/home/Datasets/DomainsInter/processed/combined_MSA_ddi_$(k)_joined_val"
# csv2fasta(pathtrain)
# csv2fasta(pathtest)
# csv2fasta(pathval)

pathfastatrain = "train_real_joined.faa"
pathfastatest = "test_real_joined.faa"
# pathfastaval = "/home/Datasets/DomainsInter/processed/combined_MSA_ddi_$(k)_joined_val_"
#learn Profile
# alignmentcc = Alignment(pathfastatrain*"2.faa", SymbolMap("ACDEFGHIKLMNPQRSTVWY-"))
# WeightsAlignment!(alignmentcc, :auto)
# profile = ProfileModel(alignmentcc; pc = 0.001)
#eval Profile
# ceprofileTrain = mean(computeCrossEntropy(profile, pathfastatrain*"2.faa"))
# ceprofileTest = mean(computeCrossEntropy(profile, pathfastatest*"2.faa"))
# ceprofileVal = mean(computeCrossEntropy(profile, pathfastaval*"2.faa"))
# fastafile = "/home/Datasets/DomainsInter/processed/combined_MSA_ddi_$(k)_joined.fasta"
arnet,arvar=ardca(pathfastatrain, verbose=false, lambdaJ=0.02,lambdaH=0.001; permorder=:NATURAL)


scoreHungarian = computeCrossEntropyPairs(arnet, "test_real_1.faa", "test_real_2.faa")
#
# temp2train = tempname()
# tempjoinedtrain = tempname()
# temp2test = tempname()
# tempjoinedtest = tempname()
# temp2val = tempname()
# tempjoinedval = tempname()
# randomPairs(pathfastatrain*"1.faa",pathfastatrain*"2.faa", temp2train, tempjoinedtrain)
# randomPairs(pathfastatest*"1.faa",pathfastatest*"2.faa", temp2test, tempjoinedtest)
# randomPairs(pathfastaval*"1.faa",pathfastaval*"2.faa", temp2val, tempjoinedval)
#
# arnetr,arvarr=ardca(tempjoinedtrain, verbose=false, lambdaJ=0.02,lambdaH=0.001; permorder=:NATURAL)
# CE_ar_train_R = computeCrossEntropy(arnetr, pathfastatrain*"1.faa", temp2train)
# CE_ar_test_R = computeCrossEntropy(arnetr, pathfastatest*"1.faa", temp2test)
# CE_ar_val_R = computeCrossEntropy(arnetr, pathfastaval*"1.faa", temp2val)
#
# @show (ceprofileTrain, ceprofileTest, ceprofileVal)
# @show (mean(CE_ar_train), mean(CE_ar_test), mean(CE_ar_val))
# @show (mean(CE_ar_train_R), mean(CE_ar_test_R), mean(CE_ar_val_R))


using NPZ
to_check = [17,46,69,71,157,160,251,258]


for k in to_check
    @show k
    pathfastatrain = "/home/Datasets/DomainsInter/processed/combined_MSA_ddi_$(k)_joined_train_"
    pathfastatest = "/home/Datasets/DomainsInter/processed/combined_MSA_ddi_$(k)_joined_test_"
    pathfastaval = "/home/Datasets/DomainsInter/processed/combined_MSA_ddi_$(k)_joined_val_"
    #learn Profile
    alignmentcc = Alignment(pathfastatrain*"2.faa", SymbolMap("ACDEFGHIKLMNPQRSTVWY-"))
    WeightsAlignment!(alignmentcc, :auto)
    arnet,arvar=ardca(pathfastatrain*"joined.faa", verbose=false, lambdaJ=0.02,lambdaH=0.001; permorder=:NATURAL)
    scoreHungarian = computeCrossEntropyPairs(arnet, pathfastaval*"1.faa", pathfastaval*"2.faa")
    npzwrite("scoH_ARDCA_$(k).npy", scoreHungarian)
end
