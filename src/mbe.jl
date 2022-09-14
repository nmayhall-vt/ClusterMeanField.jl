using ClusterMeanField

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


function compute_increment(ints, ci, fspace, rdm1a, rdm1b; verbose=2)
    #={{{=#
    @printf( "Compute increment for cluster: %s\n", string(ci))
    na = fspace[1]
    nb = fspace[2]
    no = length(ci)
    
    no_tot = n_orb(ints) 
    
    e = 0.0
    d1a = zeros(no, no)     
    d1b = zeros(no, no)     
    d1 = zeros(no, no)     
    d2 = zeros(no, no, no, no)   

    ansatz = FCIAnsatz(length(ci), na, nb)
    verbose < 0 || display(ansatz)
    ints_i = subset(ints, ci.orb_list, rdm1a, rdm1b)
    
    if ansatz.dim == 1
            #
            # we have a slater determinant. Compute energy and rdms

            if (na == no) && (nb == no)
                #
                # a doubly occupied space
                d1 = Matrix(1.0I, no, no)
                for p in 1:no, q in 1:no, r in 1:no, s in 1:no
                    d2[p,q,r,s] = 2*d1[p,q]*d1[r,s] - d1[p,s]*d1[r,q]
                end
                d1a = d1 
                d1b = d1 
                d2 *= 2.0
                e = compute_energy(0, ints_i.h1, ints_i.h2, d1, d2)
                verbose == 0 || @printf(" Slater Det Energy: %12.8f\n", e)

            elseif (na == no) && (nb == 0)
                #
                # singly occupied space
                d1 = Matrix(1.0I, no, no)
                for p in 1:no, q in 1:no, r in 1:no, s in 1:no
                    d2[p,q,r,s] = d1[p,q]*d1[r,s] - d1[p,s]*d1[r,q]
                end
                d1a  = d1
                d1b  = zeros(no,no)
                e = compute_energy(0, ints_i.h1, ints_i.h2, d1, d2)
                verbose == 0 || @printf(" Slater Det Energy: %12.8f\n", e)

            elseif (na == 0) && (nb == no)
                #
                # singly occupied space
                d1 = Matrix(1.0I, no, no)
                for p in 1:no, q in 1:no, r in 1:no, s in 1:no
                    d2[p,q,r,s] = d1[p,q]*d1[r,s] - d1[p,s]*d1[r,q]
                end
                d1a  = zeros(no,no)
                d1b  = d1
                e = compute_energy(0, ints_i.h1, ints_i.h2, d1, d2)
                verbose == 0 || @printf(" Slater Det Energy: %12.8f\n", e)

            elseif (na == 0) && (nb==0)
                # 
                # a virtual space (do nothing)
            else
                error(" How can this be?")
            end
            #e, d1, d2 = pyscf_fci(ints_i,fspace[ci.idx][1],fspace[ci.idx][2], verbose=verbose)
        else
            #
            # run PYSCF FCI
            e, d1a,d1b, d2 = pyscf_fci(ints_i,fspace[1],fspace[2], verbose=verbose)
        end
        d1 .= d1a .+ d1b
        #display(d2)
    return e, d1, d2
#=}}}=#
end

function gamma_mbe(ints::InCoreInts{T}, clusters, fspace, rdm1a, rdm1b; correct_trace=false) where T
    N = sum([length(ci) for ci in clusters])
    Nelec = sum([sum(i) for i in fspace])
    Npair = Nelec*(Nelec-1)÷2

    E1 = 0.0
    P1 = zeros(N,N)
    G1 = zeros(N,N,N,N)
    
    increments = Dict()
    increments["E"] = Dict{Tuple,T}()
    increments["P"] = Dict{Tuple,Array{T,2}}()
    increments["G"] = Dict{Tuple,Array{T,4}}()

    n_clusters = length(clusters)
    for ci in 1:n_clusters
        c = clusters[ci]
        f = (fspace[c.idx][1], fspace[c.idx][2])
        E1i, P1i, G1i = compute_increment(ints, c, f, rdm1a, rdm1b)

        increments["E"][(c,)] = E1i
        increments["P"][(c,)] = P1i
        increments["G"][(c,)] = G1i

        E1 += E1i
        P1[c.orb_list, c.orb_list] .+= P1i
        G1[c.orb_list, c.orb_list, c.orb_list, c.orb_list] .+= G1i
    end

    if correct_trace
        @printf(" Trace of G1: %12.8f\n", tr(G1))
        tmp = tr(G1) 
        for p in 1:N
            G1[p,p,p,p] = G1[p,p,p,p]*Npair/tmp
        end
        @printf(" correct...\n")
        @printf(" Trace of G1: %12.8f\n", tr(G1))
    end
    
    Enew = compute_energy(ints, P1, G1)
    @printf(" E_nuc: %12.8f\n", ints.h0)
    @printf(" 1-body quantities\n")
    @printf("   E1:     %12.8f\n", E1)
    @printf("   tr(P1): %12.8f\n", tr(P1))
    @printf("   tr(G1): %12.8f\n", tr(G1))
    @printf("   tr(HP): %12.8f\n", Enew)
   

    # 2-body
    E2 = 0.0
    P2 = zeros(N,N)
    G2 = zeros(N,N,N,N)
    n_clusters = length(clusters)
    for ci_idx in 1:n_clusters
        for cj_idx in ci_idx+1:n_clusters
            ci = clusters[ci_idx]
            cj = clusters[cj_idx]
            c = Cluster(1, [ci.orb_list..., cj.orb_list...])
            f = (fspace[ci.idx][1] + fspace[cj.idx][1], fspace[ci.idx][2] + fspace[cj.idx][2])
            E2i, P2i, G2i = compute_increment(ints, c, f, rdm1a, rdm1b)
        
            increments["E"][(ci,cj)] = E2i - increments["E"][(ci,)] - increments["E"][(cj,)]
            increments["P"][(ci,cj)] = P2i
            j_shift = length(ci)
            for (ii,i) in enumerate(ci)
                for (ii,i) in enumerate(ci)
                    increments["P"][(ci,cj)] = P2i 
                end
            end
            increments["P"][(ci,cj)][ci.orb_list, ci.orb_list] .-= increments["P"][(ci,)]
            increments["P"][(ci,cj)][cj.orb_list, cj.orb_list] .-= increments["P"][(cj,)]
            increments["G"][(ci,cj)] = G2i - increments["G"][(ci,)] - increments["G"][(cj,)]

            E2 += E2i
            P2[c.orb_list, c.orb_list] .+= P2i
            G2[c.orb_list, c.orb_list, c.orb_list, c.orb_list] .+= G2i
        end
    end

    E2 -= E1*n_clusters
    P2 -= P1*n_clusters
    G2 -= G1*n_clusters

    
    Enew = compute_energy(ints, P2, G2)
    @printf(" E_nuc: %12.8f\n", ints.h0)
    @printf(" 1-body quantities\n")
    @printf("   E2:     %12.8f\n", E2)
    @printf("   tr(P2): %12.8f\n", tr(P2))
    @printf("   tr(G2): %12.8f\n", tr(G2))
    @printf("   tr(HP): %12.8f\n", Enew)
    
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

