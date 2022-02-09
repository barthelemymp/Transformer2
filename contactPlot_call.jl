using Base.Threads: @threads, nthreads
using ArDCA
using DCAUtils
using ArgParse
using ExtractMacro: @extract
using StatsBase
using NPZ


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








function parse_commandline()
    s = ArgParseSettings()
    @add_arg_table! s begin
	"pathfastatrain"
        help = "MSA path train"
		arg_type = String
        required = true

	"pathPDB"
        help = "PDB path"
		arg_type = String
        required = true

	"chainIN"
        help = "chain for imput"
		arg_type = String
        required = true

	"chainOUT"
	    help = "PDB path"
		arg_type = String
	    required = true

	"hmmRadical"
        help = "PDB path"
		arg_type = String
        required = true

	"outputPlot"
        help = "path for plot output"
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
pathPDB = parsed_args["pathPDB"]



chainIN = parsed_args["chainIN"]
chainOUT = parsed_args["chainOUT"]
hmmRadical = parsed_args["hmmRadical"]
hmm1 = hmmRadical * "1.hmm"
hmm2 = hmmRadical * "2.hmm"
hmmjoined = hmmRadical * "joined.hmm"




write("temp2.fasta",read(`./Unalign  sample_multiple1_epoch4000_2big.faa`))
write("temp1.fasta",read(`hmmalign --outformat a2m hkrrhmmmjoined.hmm temp2.fasta`))
write("temp2.fasta",read(`./removeInserts temp1.fasta`))
pdb=PdbTool.parsePdb(pathPDB)
PdbTool.mapChainToHmm(pdb.chain[chainIN], hmm1)
PdbTool.mapChainToHmm(pdb.chain[chainOUT], hmm2)
plmo = plmdca_asym(joinpath(pwd(), "temp2.fasta"), theta = :auto)
result = makeInterRoc(plmo.score,pdb.chain[chainIN],pdb.chain[chainOUT])
