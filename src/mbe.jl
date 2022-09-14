using ClusterMeanField
using PyCall

struct Increment{T}
    #term::NTuple{N,Int}
    term::Vector{Int}
    E::Array{T,1}
    Pa::Array{T,2}
    Pb::Array{T,2}
    Gaa::Array{T,4}
    Gab::Array{T,4}
    Gbb::Array{T,4}
end
function Increment(n::Int ;T=Float64)
    return Increment{T}(Vector{Int}([]), [0.0], zeros(n,n), zeros(n,n), zeros(n,n,n,n), zeros(n,n,n,n), zeros(n,n,n,n))
end
function Increment(da::Array{T,2}, db::Array{T,2})  where {T}
    n = size(da,1)
    n == size(da,2) || throw(DimensionMismatch)
    n == size(db,1) || throw(DimensionMismatch)
    n == size(db,2) || throw(DimensionMismatch)
    incr = Increment{T}([], [0.0], da, db, zeros(n,n,n,n), zeros(n,n,n,n), zeros(n,n,n,n))
    Gaa = incr.Gaa
    Gab = incr.Gab
    Gbb = incr.Gbb
    @tensor begin
        Gaa[p,q,r,s] += da[p,q] * da[r,s]
        Gaa[p,q,r,s] -= da[p,s] * da[r,q]

        Gbb[p,q,r,s] += db[p,q] * db[r,s]
        Gbb[p,q,r,s] -= db[p,s] * db[r,q]

        Gab[p,q,r,s] += da[p,q] * db[r,s]
    end
    incr.Gaa .= Gaa
    incr.Gab .= Gab
    incr.Gbb .= Gbb
    return incr
end
function Base.display(incr::Increment)
    @printf(" :Increment:   %s\n",incr.term)
    @printf("   E:          %12.8f\n",incr.E[1])
    @printf("   tr(Pa):     %12.8f\n",tr(incr.Pa))
    @printf("   tr(Pb):     %12.8f\n",tr(incr.Pb))
    @printf("   tr(Gaa):    %12.8f\n",tr(incr.Gaa))
    @printf("   tr(Gab):    %12.8f\n",tr(incr.Gab))
    @printf("   tr(Gbb):    %12.8f\n",tr(incr.Gbb))
end

function InCoreIntegrals.compute_energy(ints::InCoreInts, incr::Increment)
    e = ints.h0
    e += sum(ints.h1 .* incr.Pa)
    e += sum(ints.h1 .* incr.Pb)
    e += .5*sum(ints.h2 .* incr.Gaa)
    e += .5*sum(ints.h2 .* incr.Gbb)
    e += sum(ints.h2 .* incr.Gab)
    return e
end


"""
"""
function compute_energy2(h0, h1, h2, rdms::Dict{String, Array{T}}) where T
    length(rdms["Pa"]) == length(h1) || throw(DimensionMismatch)
    length(rdms["Pb"]) == length(h1) || throw(DimensionMismatch)
    length(rdms["Gaa"]) == length(h2) || throw(DimensionMismatch)
    length(rdms["Gbb"]) == length(h2) || throw(DimensionMismatch)
    length(rdms["Gab"]) == length(h2) || throw(DimensionMismatch)

    e = h0
    e += sum(h1 .* rdms["Pa"])
    e += sum(h1 .* rdms["Pb"])
    e += .5*sum(h2 .* rdms["Gaa"])
    e += .5*sum(h2 .* rdms["Gbb"])
    e += sum(h2 .* rdms["Gab"])
    return e
end
"""
"""
function compute_energy2(ints::InCoreInts, rdms::Dict{String, Array{T}}) where T
    return compute_energy2(ints.h0, ints.h1, ints.h2, rdms)
end
"""
    compute_energy(ints::InCoreInts, rdm1, rdm2)

Return energy defined by `rdm1` and `rdm2` which are not spin-traced 
1 and 2 RDMs
"""
function compute_energy2(ints::InCoreInts, rdm1::Dict{String, Array{T,2}}, rdm2::Dict{String, Array{T,4}}) where T
    return compute_energy2(ints.h0, ints.h1, ints.h2, rdm1, rdm2) 
end

"""
    compute_energy(ints::InCoreInts, rdm1, rdm2)

Return energy defined by `rdm1` and `rdm2` which are not spin-traced 
1 and 2 RDMs
"""
function compute_energy2(h0, h1, h2, rdm1::Dict{String, Array{T,2}}, rdm2::Dict{String, Array{T,4}}) where T
    length(rdm1["a"]) == length(h1) || throw(DimensionMismatch)
    length(rdm1["b"]) == length(h1) || throw(DimensionMismatch)
    length(rdm2["aa"]) == length(h2) || throw(DimensionMismatch)
    length(rdm2["bb"]) == length(h2) || throw(DimensionMismatch)
    length(rdm2["ab"]) == length(h2) || throw(DimensionMismatch)

    e = h0
    e += sum(h1 .* rdm1["a"])
    e += sum(h1 .* rdm1["b"])
    e += .5*sum(h2 .* rdm2["aa"])
    e += .5*sum(h2 .* rdm2["bb"])
    e += sum(h2 .* rdm2["ab"])
    return e
end

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

function compute_increment(ints::InCoreInts{T}, ci, fspace, rdm1a, rdm1b; 
                           verbose=2, max_cycle=100, conv_tol=1e-8) where T
    #={{{=#
    @printf( "Compute increment for cluster: %s\n", string(ci))
    na = fspace[1]
    nb = fspace[2]
    no = length(ci)

    no_tot = n_orb(ints) 

    e = 0.0

    ansatz = FCIAnsatz(length(ci), na, nb)
    verbose < 0 || display(ansatz)
    ints_i = subset(ints, ci.orb_list, rdm1a, rdm1b)

    if ansatz.dim == 1
        #
        # we have a slater determinant. Compute energy and rdms

        if (na == no) && (nb == no)
            #
            # doubly occupied space
            da = Matrix(1.0I, no, no)
            db = Matrix(1.0I, no, no)
            Gaa = zeros(no, no, no, no) 
            Gab = zeros(no, no, no, no) 
            Gbb = zeros(no, no, no, no) 
            for p in 1:no, q in 1:no, r in 1:no, s in 1:no
                Gaa[p,q,r,s] = da[p,q]*da[r,s] - da[p,s]*d1["a"][r,q]
                Gbb[p,q,r,s] = db[p,q]*db[r,s] - db[p,s]*d1["b"][r,q]
                Gab[p,q,r,s] = da[p,q]*db[r,s]
            end
            display(ints_i.h0)
            e = compute_energy(ints_i, da + db, Gaa + 2*Gab + Gbb)
            verbose == 0 || @printf(" Slater Det Energy: %12.8f\n", e)

        elseif (na == no) && (nb == 0)

        elseif (na == 0) && (nb == no)

        elseif (na == 0) && (nb==0)
            # 
            # a virtual space (do nothing)
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
        e, vfci = cisolver.kernel(ints_i.h1, ints_i.h2, no, (na,nb), ecore=0)
        (d1["a"], d1["b"]), (d2["aa"], d2["ab"], d2["bb"])  = cisolver.make_rdm12s(vfci, no, (na,nb))
        display(tr(d1["a"]))
        display(tr(d1["b"]))
        display(tr(d2["aa"]))
        display(tr(d2["ab"]))
        display(tr(d2["bb"]))
    end
    return e, d1, d2
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
    Nelec = sum([sum(i) for i in fspace])
    Npair = Nelec*(Nelec-1)÷2

    ref_data = Increment(rdm1a, rdm1b)
    E0 = compute_energy(ints, ref_data)
    ref_data.E .= E0

    display(ref_data)

    # Start by just doing everything int the full space (dumb)

    increments = Vector{Increment{T}}([])

    n_clusters = length(clusters)
    for ci in 1:n_clusters
        c = clusters[ci]
        f = (fspace[c.idx][1], fspace[c.idx][2])
        E1i, P1i, G1i = compute_increment(ints, c, f, rdm1a, rdm1b)
        
        E1 = 0.0
        P1a = zeros(N,N)
        P1b = zeros(N,N)
        G1aa = zeros(N,N,N,N)
        G1ab = zeros(N,N,N,N)
        G1bb = zeros(N,N,N,N)

        E1 += E1i 
        P1a[c.orb_list, c.orb_list] .+= P1i["a"] - ref_data.Pa[c.orb_list, c.orb_list]
        P1b[c.orb_list, c.orb_list] .+= P1i["b"] - ref_data.Pa[c.orb_list, c.orb_list]
        G1aa[c.orb_list, c.orb_list, c.orb_list, c.orb_list] .+= G1i["aa"] - ref_data.Gaa[c.orb_list, c.orb_list, c.orb_list, c.orb_list]
        G1ab[c.orb_list, c.orb_list, c.orb_list, c.orb_list] .+= G1i["ab"] - ref_data.Gab[c.orb_list, c.orb_list, c.orb_list, c.orb_list]
        G1bb[c.orb_list, c.orb_list, c.orb_list, c.orb_list] .+= G1i["bb"] - ref_data.Gbb[c.orb_list, c.orb_list, c.orb_list, c.orb_list]

        push!(increments, Increment([ci], [E1], P1a, P1b, G1aa, G1ab, G1bb))
    end


    # 2-body
    nbody = 1
    if nbody > 1 
        for ci_idx in 1:n_clusters
            for cj_idx in ci_idx+1:n_clusters
                ci = clusters[ci_idx]
                cj = clusters[cj_idx]
                c = Cluster(1, [ci.orb_list..., cj.orb_list...])
                f = (fspace[ci.idx][1] + fspace[cj.idx][1], fspace[ci.idx][2] + fspace[cj.idx][2])
                Ei, Pi, Gi = compute_increment(ints, c, f, rdm1a, rdm1b)

                Eincr = 0.0
                Pincra = zeros(N,N)
                Pincrb = zeros(N,N)
                Gincraa = zeros(N,N,N,N)
                Gincrab = zeros(N,N,N,N)
                Gincrbb = zeros(N,N,N,N)

                Eincr += Ei 
                Eincr -= increments["E"][(ci,)] + increments["E"][(cj,)]

                Pincra[c.orb_list, c.orb_list] .+= Pi["a"]
                Pincra .-=  increments["Pa"][(ci,)] + increments["Pa"][(cj,)]

                Gincraa[c.orb_list, c.orb_list, c.orb_list, c.orb_list] .+= Gi["aa"]
                Gincraa .-=  increments["Gaa"][(ci,)] + increments["Gaa"][(cj,)]

                increments["E"][(ci,cj)] = Eincr
                increments["Pa"][(ci,cj)] = Pincra 
                increments["Pb"][(ci,cj)] = Pincrb
                increments["Gaa"][(ci,cj)] = Gincraa
                increments["Gab"][(ci,cj)] = Gincrab
                increments["Gbb"][(ci,cj)] = Gincrbb

            end
        end
    end

    @printf(" Add results\n")
    E = 0.0
    Pa = zeros(N,N)
    Pb = zeros(N,N)
    Gaa = zeros(N,N,N,N)
    Gab = zeros(N,N,N,N)
    Gbb = zeros(N,N,N,N)
    final_data = Increment(N) 
    display(ref_data)
    for inc in increments
        display(inc)
        final_data.E .+= inc.E
        final_data.Pa .+= inc.Pa
        final_data.Pb .+= inc.Pb
        final_data.Gaa .+= inc.Gaa
        final_data.Gab .+= inc.Gab
        final_data.Gbb .+= inc.Gbb
    end
    

    Enew = compute_energy(ints, final_data)
    @printf(" E_nuc: %12.8f\n", ints.h0)
    @printf(" Summed quantities\n")
    @printf("   E0:      %12.8f\n", E0)
    @printf("   tr(HG):  %12.8f\n", Enew)
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

