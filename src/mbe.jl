using ClusterMeanField
using PyCall



"""
    subset(ints::InCoreInts, list, rmd1a, rdm1b)

Extract a subset of integrals acting on orbitals in list, returned as `InCoreInts` type
and contract a 1rdm to give effectve 1 body interaction

# Arguments
- `ints::InCoreInts`: Integrals for full system 
- `list`: list of orbital indices in subset
- `rdm1a`: 1RDM for embedding α density to make CASCI hamiltonian
- `rdm1b`: 1RDM for embedding β density to make CASCI hamiltonian
"""
function InCoreIntegrals.subset(ints::InCoreInts, ci::Cluster, rdm1a, rdm1b)
    list = ci.orb_list
    ints_i = subset(ints, list)
    da = deepcopy(rdm1a)
    db = deepcopy(rdm1b)
    da[:,list] .= 0
    db[:,list] .= 0
    da[list,:] .= 0
    db[list,:] .= 0
    viirs = ints.h2[list, list,:,:]
    viqri = ints.h2[list, :, :, list]
    f = zeros(length(list),length(list))
    @tensor begin
        f[p,q] += viirs[p,q,r,s] * (da+db)[r,s]
        f[p,s] -= .5*viqri[p,q,r,s] * da[q,r]
        f[p,s] -= .5*viqri[p,q,r,s] * db[q,r]
    end
    ints_i.h1 .+= f
    h0 = compute_energy(ints, (da,db))
    return InCoreInts(h0, ints_i.h1, ints_i.h2) 
end

"""
    compute_energy(ints::InCoreInts, rdm1::Tuple{Matrix, Matrix})

Return energy defined by `rdm1`. rdm1 is a tuple for the alpha and beta
density matrices respectively.
"""
function InCoreIntegrals.compute_energy(ints::InCoreInts, rdm1::Tuple{Matrix,Matrix})

    length(rdm1[1]) == length(ints.h1) || throw(DimensionMismatch)
    length(rdm1[2]) == length(ints.h1) || throw(DimensionMismatch)
    
    e = ints.h0
    @tensor begin
        e += ints.h1[p,q] * rdm1[1][p,q]
        e += ints.h1[p,q] * rdm1[2][p,q]
       
        #aa
        e += .5 * ints.h2[p,q,r,s] * rdm1[1][p,q] * rdm1[1][r,s]
        e -= .5 * ints.h2[p,q,r,s] * rdm1[1][p,s] * rdm1[1][r,q]
        
        #bb
        e += .5 * ints.h2[p,q,r,s] * rdm1[2][p,q] * rdm1[2][r,s]
        e -= .5 * ints.h2[p,q,r,s] * rdm1[2][p,s] * rdm1[2][r,q]
        
        e += ints.h2[p,q,r,s] * rdm1[1][p,q] * rdm1[2][r,s]
    end
    return e
end


function compute_fock(ints::InCoreInts, rdm1::Tuple{Matrix,Matrix})
#={{{=#
    fa = deepcopy(ints.h1)
    fb = deepcopy(ints.h1)
    @tensor begin
        #a
        fa[r,s] += 0.5 * ints.h2[p,q,r,s] * rdm1[1][p,q] 
        fa[r,s] -= 0.5 * ints.h2[p,r,q,s] * rdm1[1][p,q]
        fa[r,s] += 0.5 * ints.h2[p,q,r,s] * rdm1[2][p,q] 
        
        #b
        fb[r,s] += 0.5 * ints.h2[p,q,r,s] * rdm1[2][p,q] 
        fb[r,s] -= 0.5 * ints.h2[p,r,q,s] * rdm1[2][p,q]
        fb[r,s] += 0.5 * ints.h2[p,q,r,s] * rdm1[1][p,q]
        
    end
    return (fa,fb) 
end
#=}}}=#

function LinearAlgebra.tr(A::Array{T,4}) where T
#={{{=#
    N = size(A,1)
    N == size(A,2) || throw(DimensionMismatch)
    N == size(A,3) || throw(DimensionMismatch)
    N == size(A,4) || throw(DimensionMismatch)
    out = T(0)
    for i in 1:N
        out += A[i,i,i,i]
    end
    return out
end
#=}}}=#

"""
    spin_trace(rdm1::Dict{String,Array{T,2}}, rdm2::Dict{String,Array{T,4}})

Integrate out spin from our RDMs

P = Pa + Pb
G = Gaa + Gbb + 2Gab
"""
function spin_trace(rdm1::Dict{String,Array{T,2}}, rdm2::Dict{String,Array{T,4}}) where T
    return rdm1["a"] .+ rdm1["b"], rdm2["aa"] .+ 2 .* rdm2["ab"] .+ rdm2["bb"]
end

function compute_increment(ints::InCoreInts{T}, cluster_set::Vector{Cluster}, fspace, rdm1a, rdm1b; 
                           verbose=0, max_cycle=100, conv_tol=1e-8) where T
    #={{{=#
    @printf( "\n*Compute increment for cluster:     ")
    [@printf("%3i",c.idx)  for c in cluster_set]
    println()
    na = 0
    nb = 0
    no = 0
    
    orb_list = []
    for ci in cluster_set
        na += fspace[ci.idx][1]
        nb += fspace[ci.idx][2]
        no += length(ci)
        append!(orb_list, ci.orb_list)
    end

    ci = Cluster(0, orb_list)   # this is our cluster for this increment

    out = Increment(cluster_set)
    no_tot = n_orb(ints) 

    e = 0.0

    ansatz = FCIAnsatz(length(ci), na, nb)
    verbose < 0 || display(ansatz)
    ints_i = subset(ints, ci.orb_list, rdm1a, rdm1b)
    ints_i = subset(ints, ci, rdm1a, rdm1b)
    #ints_i = form_1rdm_dressed_ints(ints, ci.orb_list, rdm1a, rdm1b)
        
    d1a = rdm1a[ci.orb_list, ci.orb_list] 
    d1b = rdm1b[ci.orb_list, ci.orb_list] 
    d2aa = zeros(no, no, no, no) 
    d2ab = zeros(no, no, no, no) 
    d2bb = zeros(no, no, no, no) 

    if ansatz.dim == 1
        #
        # we have a slater determinant. Compute energy and rdms

        if (na == no) && (nb == no)
            #
            # doubly occupied space
            for p in 1:no, q in 1:no, r in 1:no, s in 1:no
                d2aa[p,q,r,s] = d1a[p,q]*d1a[r,s] - d1a[p,s]*d1a[r,q]
                d2bb[p,q,r,s] = d1b[p,q]*d1b[r,s] - d1b[p,s]*d1b[r,q]
                d2ab[p,q,r,s] = 2*d1a[p,q]*d1b[r,s]
            end
            #e = compute_energy(ints_i, da + db, Gaa + 2*Gab + Gbb)
            #verbose == 0 || @printf(" Slater Det Energy: %12.8f\n", e)
            e = compute_energy(ints_i, (d1a, d1b))
            verbose == 0 || @printf(" Slater Det Energy: %12.8f\n", e)

        elseif (na == no) && (nb == 0)
                #
                # singly occupied space
                for p in 1:no, q in 1:no, r in 1:no, s in 1:no
                    d2aa[p,q,r,s] = d1a[p,q]*d1a[r,s] - d1a[p,s]*d1a[r,q]
                end
                e = compute_energy(ints, (d1a, d1b))
                #e = compute_energy(0, ints_i.h1, ints_i.h2, d1, d2)
                verbose == 0 || @printf(" Slater Det Energy: %12.8f\n", e)

        elseif (na == 0) && (nb == no)
                #
                # singly occupied space
                for p in 1:no, q in 1:no, r in 1:no, s in 1:no
                    d2bb[p,q,r,s] = d1b[p,q]*d1b[r,s] - d1b[p,s]*d1b[r,q]
                end
                e = compute_energy(ints, (d1a, d1b))
                #e = compute_energy(0, ints_i.h1, ints_i.h2, d1, d2)
                verbose == 0 || @printf(" Slater Det Energy: %12.8f\n", e)

        elseif (na == 0) && (nb==0)
            # 
            # a virtual space (do nothing)
            e = compute_energy(ints_i, (d1a, d1b))
            verbose == 0 || @printf(" Slater Det Energy: %12.8f\n", e)
        else
            error(" How can this be?")
        end
    else
        #
        # run PYSCF FCI
        #e, d1a,d1b, d2 = pyscf_fci(ints_i,fspace[1],fspace[2], verbose=verbose)
        pyscf = pyimport("pyscf")
        fci = pyimport("pyscf.fci")
        cisolver = pyscf.fci.direct_spin1.FCI()
        cisolver.max_cycle = max_cycle
        cisolver.conv_tol = conv_tol
        nelec = na + nb
        e, vfci = cisolver.kernel(ints_i.h1, ints_i.h2, no, (na,nb), ecore=ints_i.h0)
        (d1a, d1b), (d2aa, d2ab, d2bb)  = cisolver.make_rdm12s(vfci, no, (na,nb))
        println(e)
    end
      
    out.E .= [e]
    out.Pa  .= d1a
    out.Pb  .= d1b
    out.Gaa .= d2aa
    out.Gab .= d2ab
    out.Gbb .= d2bb

    return out 
    #=}}}=#
end

"""
    build_cumulant(rdm2, rdm1)

take in rdms and construct cumulant
# Arguments:
- `rdm2`: 2rdm ordered [1,1,2,2] such that E+= 1/2 (p,q|r,s)2rdm[p,q,r,s]
- `rdm1`: 1rdm 
"""
function build_cumulant(rdm2, rdm1)

    c = zeros(size(rdm2))
    @tensor begin
        c[p,q,r,s] = rdm2[p,q,r,s] - rdm1[p,q]*rdm1[r,s] + rdm1[p,s]*rdm1[q,r]
    end
end

function gamma_mbe(ints::InCoreInts{T}, clusters, fspace, rdm1a, rdm1b; verbose=1) where T
    N = sum([length(ci) for ci in clusters])
    N == size(rdm1a,1) || throw(DimensionMismatch)
    
    Nelec = sum([sum(i) for i in fspace])
    Npair = Nelec*(Nelec-1)÷2

    ref_data = Increment([Cluster(0, [i for i in 1:N])], rdm1a, rdm1b)
    E0 = compute_energy(ints, ref_data)
    ref_data.E .= E0

    println(" Nick")
    display(compute_energy(ints, (rdm1a, rdm1b)))
    display(compute_energy(ints, ref_data))
    display(ref_data)


    # Start by just doing everything int the full space (dumb)

    increments = Dict{Tuple,Increment{T}}()

    n_clusters = length(clusters)
    for i in 1:n_clusters
        ci = clusters[i]
        out_i = compute_increment(ints, [ci], fspace, rdm1a, rdm1b)
        
        # subtract the hartree fock data so that we have an increment from HF
        out_i = out_i - ref_data
        display(out_i)
        increments[(i,)] = out_i
    end


    # 2-body
    nbody = 2
    if nbody > 1 
        for i in 1:n_clusters
            for j in i+1:n_clusters
                ci = clusters[i]
                cj = clusters[j]
                out_i = compute_increment(ints, [ci, cj], fspace, rdm1a, rdm1b)

                # subtract the hartree fock data so that we have an increment from HF
                out_i = out_i - ref_data - increments[(i,)] - increments[(j,)]
                increments[(i,j)] = out_i
                display(out_i) 
            end
        end
    end

    @printf(" Add results\n")
    final_data = deepcopy(ref_data) 
    for (key,inc) in increments
        display(inc)
        final_data = final_data + inc
    end
    

    Enew = compute_energy(ints, final_data)
    @printf(" Summed quantities\n")
    @printf("   E0:      %12.8f\n", E0)
    @printf("   tr(HG):  %12.8f\n", Enew)
    println(" Input data:")
    display(ref_data)
    println(" Output data:")
    display(final_data)
   
end



#"""
#    assemble_full_rdms(clusters::Vector{Cluster}, rdm1s::Dict{Integer, Array}, rdm2s::Dict{Integer, Array})
#Return 1 and 2 RDMs
#"""
#function assemble_full_rdms(clusters::Vector{Cluster}, rdm1s::Dict{Integer, Array}, rdm2s::Dict{Integer, Array})
#    norb = sum([length(i) for i in clusters])
#    @printf(" Norbs: %i\n",norb)
#    rdm1a = zeros(norb,norb)
#    rdm1b = zeros(norb,norb)
#    for ci in clusters
#        rdm1a[ci.orb_list, ci.orb_list] .= rdm1s[ci.idx][1]
#        rdm1b[ci.orb_list, ci.orb_list] .= rdm1s[ci.idx][2]
#    end
#
#    rdm2aa = zeros(norb,norb,norb,norb)
#    rdm2bb = zeros(norb,norb,norb,norb)
#    rdm2ab = zeros(norb,norb,norb,norb)
#    @tensor begin
#        rdm2aa[p,q,r,s] += rdm1a[p,q] * rdm1a[r,s]
#        rdm2aa[p,q,r,s] -= rdm1a[p,s] * rdm1a[r,q]
#
#        rdm2bb[p,q,r,s] += rdm1b[p,q] * rdm1b[r,s]
#        rdm2bb[p,q,r,s] -= rdm1b[p,s] * rdm1b[r,q]
#
#        rdm2ab[p,q,r,s] += rdm1a[p,q] * rdm1b[r,s]
#    end
#
#    for ci in clusters
#        rdm2aa[ci.orb_list, ci.orb_list, ci.orb_list, ci.orb_list] .= rdm2s[ci.idx][1]
#        rdm2bb[ci.orb_list, ci.orb_list, ci.orb_list, ci.orb_list] .= rdm2s[ci.idx][2]
#        rdm2ab[ci.orb_list, ci.orb_list, ci.orb_list, ci.orb_list] .= rdm2s[ci.idx][3]
#    end
#    return (rdm1a, rdm1b), (rdm2aa, rdm2bb, rdm2ab)
#end

