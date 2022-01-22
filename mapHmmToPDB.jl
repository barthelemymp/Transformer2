using PdbTool
using ArgParse
using DelimitedFiles

using Statistics
using ArgParse

function parse_commandline()
    s = ArgParseSettings()
    @add_arg_table! s begin
	"msaPath"
        help = "MSA path"
		arg_type = String
        required = true

	"pdbPath"
        help = "PDB path"
		arg_type = String
        required = true

	"pdbChain"
        help = "PDB chain ID"
		arg_type = String
        required = true

	"csvOUT"
        help = "where to save csvoutput"
		arg_type = String
        required = true
    end
    return parse_args(s)
end

function get_HMMlength(HmmPath)
	a = String(read(`grep LENG $HmmPath`))
	b = split(a, "  ")[2]
	c = split(b, "\n")[1]
	return parse(Int64,c)
end

function averagePosition(res::PdbTool.Residue)
	x = 0.0
	y = 0.0
	z = 0.0
	count = 0
	for atom1 in values(res.atom)
		x+=atom1.coordinates[1]
		y+=atom1.coordinates[2]
		z+=atom1.coordinates[3]
		count+=1
	end
	x = x/count
	y = y/count
	z = z/count
	return (x,y,z)
end

parsed_args = parse_commandline()
println("Parsed args:")
for (arg,val) in parsed_args
    println("  $arg  =>  $val")
end

msaPath = parsed_args["msaPath"]
pdbPath = parsed_args["pdbPath"]
pdbChain = parsed_args["pdbChain"]
csvOUT = parsed_args["csvOUT"]

tempFileHMM=tempname()

run(`hmmbuild --symfrac 0.0 $tempFileHMM $msaPath`)
L = get_HMMlength(tempFileHMM)

pdb=PdbTool.parsePdb(pdbPath)
PdbTool.mapChainToHmm(pdb.chain[pdbChain],"$tempFileHMM")
chainresidue = [i for i in 1:10000 if "$i" in keys(pdb.chain[pdbChain].residue)]

pdbSeq=PdbTool.chainSeq(pdb.chain[pdbChain])
hmmout = PdbTool.mapSeqToHmm(pdbSeq,tempFileHMM)

alignout  = split(PdbTool.alignSeqToHmm(pdbSeq,tempFileHMM), "")

DataCSV = Array{Float64}(undef, (L+2,3))

Notmapped = []
mapped = []
for i in 2:L+1
	if i in keys(hmmout)
		res = pdb.chain[pdbChain].residue["$(chainresidue[hmmout[i]])"]
		pos = averagePosition(res)
		DataCSV[i,1] = pos[1]
		DataCSV[i,2] = pos[2]
		DataCSV[i,3] = pos[3]
		push!(mapped, i)
	else
		DataCSV[i,1] = 0.0
		DataCSV[i,2] = 0.0
		DataCSV[i,3] = 0.0
		push!(Notmapped, i)
	end
end


for i in Notmapped
	mappeddist = abs.(copy(mapped) .-i)
	closest = mapped[findmin(mappeddist)[2]]
	@show i, closest
	DataCSV[i,1] = DataCSV[closest,1]
	DataCSV[i,2] = DataCSV[closest,2]
	DataCSV[i,3] = DataCSV[closest,3]
end






DataCSV[1,1] = 0.0
DataCSV[1,2] = 0.0
DataCSV[1,3] = 0.0
DataCSV[L+2,1] = 0.0
DataCSV[L+2,2] = 0.0
DataCSV[L+2,3] = 0.0

DataCSV[:,1] .= DataCSV[:,1] .-  mean(DataCSV[:,1])
DataCSV[:,1] .= DataCSV[:,1]./maximum(abs.(DataCSV[:,1])) *L

DataCSV[:,2] .= DataCSV[:,2] .-  mean(DataCSV[:,2])
DataCSV[:,2] .= DataCSV[:,2]./maximum(abs.(DataCSV[:,2])) *L

DataCSV[:,3] .= DataCSV[:,3] .-  mean(DataCSV[:,3])
DataCSV[:,3] .= DataCSV[:,3]./maximum(abs.(DataCSV[:,3])) *L


DataCSV[:,2] .-= mean(DataCSV[:,2])
DataCSV[:,3] .-= mean(DataCSV[:,3])
@show pwd()

writedlm(joinpath(pwd(),csvOUT),  DataCSV, ',')
rm("$tempFileHMM")



#
#
#
# 		DataCSV = Array{String}(undef, (Nexemples,2))
# 		DataCSV[i,1] = join(seq1, " ")
# 		pp = "/home/Datasets/DomainsInter/PPIprocessed/PPI_$(k)_joined.csv"
#         writedlm(pp,  DataCSV, ',')
#
#
# chainresidue = [i for i in 1:10000 if "$i" in keys(pdb.chain[pdbChain].residue)]
# ch.residue["$(chainresidue[hmmout[17]])"]
#
# for pair in 1:length(orderedPairs)
# 	i = orderedPairs[pair][1]
# 	j = orderedPairs[pair][2]
# 	i_residue = chainresidue[i]
# 	j_residue = chainresidue[j]
# 	dist = PdbTool.residueDist(chain.residue["$i_residue"], chain.residue["$j_residue"])
#
# using JSON
#
# people = [Dict("name"=>"CoolGuy", "company"=>"tech") for i=1:1000]
# companies = [Dict("name"=>"CoolTech", "address"=>"Bay Area") for i=1:100]
#
# data = Dict("people"=>people, "companies"=>companies)
# json_string = JSON.json(data)
#
# open("foo.json","w") do f
#   JSON.print(f, json_string)
# end
#
