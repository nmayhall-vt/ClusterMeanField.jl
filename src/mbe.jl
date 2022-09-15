using ClusterMeanField
using PyCall


"""
    screen(nodes, thresh, increments)

Decide if a term should be computed by determining if the 
2body correlation graph is connected. 
Here we screen on the eigenvalue of the fiedler vector.

# Note
In order to use this, we will need to modify MBE to allow for 
terms to be missing from the increments dictionary.
"""
function screen(nodes, thresh, increments)
    N = length(nodes)
    L = zeros(N,N)
    for ii in 1:N
        for jj in ii+1:N
            i = nodes[ii]
            j = nodes[jj]
            L[ii,jj] = -abs(increments[(i,j)].E[1])
            L[jj,ii] = L[ii,jj]
        end
    end
    for i in 1:N
        L[i,i] = -sum(L[:,i])
    end
    l,_ = eigen(L)
    l = sort(l) 
    if l[2] < thresh
        return true     # skip
    else 
        return false    # compute
    end
end

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
    for j in 1:N
        out += A[i,i,j,j]
    end
    end
    return out
end
#=}}}=#


function compute_increment(ints::InCoreInts{T}, cluster_set::Vector{Cluster}, fspace, rdm1a, rdm1b; 
                           verbose=0, max_cycle=100, conv_tol=1e-8, screen=1e-12) where T
    #={{{=#
    verbose < 1 || @printf( "\n*Compute increment for cluster:     ")
    verbose < 1 || [@printf("%3i",c.idx)  for c in cluster_set]
    verbose < 1 || println()
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
    verbose < 1 || display(ansatz)
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
            (d2aa, d2ab, d2bb) = build_2rdm((d1a, d1b))
            e = compute_energy(ints_i, (d1a, d1b))
            verbose < 2 || @printf(" Slater Det Energy: %12.8f\n", e)

        elseif (na == no) && (nb == 0)
                #
                # singly occupied space
                for p in 1:no, q in 1:no, r in 1:no, s in 1:no
                    d2aa[p,q,r,s] = d1a[p,q]*d1a[r,s] - d1a[p,s]*d1a[r,q]
                end
                e = compute_energy(ints, (d1a, d1b))
                #e = compute_energy(0, ints_i.h1, ints_i.h2, d1, d2)
                verbose < 2 || @printf(" Slater Det Energy: %12.8f\n", e)

        elseif (na == 0) && (nb == no)
                #
                # singly occupied space
                for p in 1:no, q in 1:no, r in 1:no, s in 1:no
                    d2bb[p,q,r,s] = d1b[p,q]*d1b[r,s] - d1b[p,s]*d1b[r,q]
                end
                e = compute_energy(ints, (d1a, d1b))
                #e = compute_energy(0, ints_i.h1, ints_i.h2, d1, d2)
                verbose < 2 || @printf(" Slater Det Energy: %12.8f\n", e)

        elseif (na == 0) && (nb==0)
            # 
            # a virtual space (do nothing)
            e = compute_energy(ints_i, (d1a, d1b))
            verbose < 2|| @printf(" Slater Det Energy: %12.8f\n", e)
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


    end
      
    out.E .= [e]
    out.Da  .= d1a
    out.Db  .= d1b
    out.Daa .= d2aa
    out.Dab .= d2ab
    out.Dbb .= d2bb


    (tmpaa, tmpab, tmpbb) = build_2rdm((d1a, d1b))
    out.Caa .= d2aa .- tmpaa
    out.Cab .= d2ab .- tmpab
    out.Cbb .= d2bb .- tmpbb

    return out 
    #=}}}=#
end

"""
    build_2rdm((Da,Db))

take in rdms and construct 2rdm
# Arguments:
- `rdm1`: tuple of 1rdm's (alpha, beta)

# Returns:
- `Tuple` of 2rdms, (aa, ab, bb) spin cases.
"""
function build_2rdm((Da,Db))

    no = size(Da,1)
    Daa = zeros(no, no, no, no)
    Dab = zeros(no, no, no, no)
    Dbb = zeros(no, no, no, no)
    for p in 1:no, q in 1:no, r in 1:no, s in 1:no
        Daa[p,q,r,s] = Da[p,q]*Da[r,s] - Da[p,s]*Da[r,q]
        Dbb[p,q,r,s] = Db[p,q]*Db[r,s] - Db[p,s]*Db[r,q]
        Dab[p,q,r,s] = Da[p,q]*Db[r,s]
    end
    return (Daa, Dab, Dbb)
end

function gamma_mbe(nbody, ints::InCoreInts{T}, clusters, fspace, rdm1a, rdm1b; verbose=1, thresh=1e-12) where T
    N = sum([length(ci) for ci in clusters])
    N == size(rdm1a,1) || throw(DimensionMismatch)
    
    Nelec = sum([sum(i) for i in fspace])
    Npair = Nelec*(Nelec-1)÷2

    ref_data = Increment([Cluster(0, [i for i in 1:N])], rdm1a, rdm1b)
    E0 = compute_energy(ints, ref_data)
    ref_data.E .= E0

    println(" Reference Data")
    display(ref_data)


    # Start by just doing everything int the full space (dumb)

    increments = Dict{Tuple,Increment{T}}()

    n_clusters = length(clusters)
    # 1-body
    if nbody >= 1
        for i in 1:n_clusters
            ci = clusters[i]
            out_i = compute_increment(ints, [ci], fspace, rdm1a, rdm1b, verbose=verbose)

            # subtract the hartree fock data so that we have an increment from HF
            out_i = out_i - ref_data
            increments[(i,)] = out_i
            
            verbose < 2 || display(out_i)
        end
    end


    # 2-body
    if nbody >= 2 
        for i in 1:n_clusters
            for j in i+1:n_clusters
                
                ci = clusters[i]
                cj = clusters[j]
                out_i = compute_increment(ints, [ci, cj], fspace, rdm1a, rdm1b, verbose=verbose)

                # subtract the hartree fock data so that we have an increment from HF
                out_i = out_i - ref_data
                
                out_i = out_i - increments[(i,)]
                out_i = out_i - increments[(j,)]

                increments[(i,j)] = out_i
                verbose < 2 || display(out_i) 
            end
        end
    end

    # 3-body
    if nbody >= 3 
        for i in 1:n_clusters
            for j in i+1:n_clusters
                for k in j+1:n_clusters
                    ci = clusters[i]
                    cj = clusters[j]
                    ck = clusters[k]

                    #screen((i, j, k), thresh, increments) == false || continue

                    out_i = compute_increment(ints, [ci, cj, ck], fspace, rdm1a, rdm1b, verbose=verbose)

                    # subtract the hartree fock data so that we have an increment from HF
                    out_i = out_i - ref_data

                    out_i = out_i - increments[(i,j)]
                    out_i = out_i - increments[(i,k)]
                    out_i = out_i - increments[(j,k)]
                    
                    out_i = out_i - increments[(i,)]
                    out_i = out_i - increments[(j,)]
                    out_i = out_i - increments[(k,)]

                    increments[(i,j,k)] = out_i
                    verbose < 2 || display(out_i)
                end
            end
        end
    end

    # 4-body
    if nbody >= 4 
        for i in 1:n_clusters
            for j in i+1:n_clusters
                for k in j+1:n_clusters
                    for l in k+1:n_clusters
                    
                        #screen((i, j, k, l), thresh, increments) == false || continue
                    
                        ci = clusters[i]
                        cj = clusters[j]
                        ck = clusters[k]
                        cl = clusters[l]
                        out_i = compute_increment(ints, [ci, cj, ck, cl], fspace, rdm1a, rdm1b, verbose=verbose)

                        # subtract the hartree fock data so that we have an increment from HF
                        subtract!(out_i, ref_data) 
                        
                        # subtract the lower orders 
                        subtract!(out_i, increments[(i,j,k)])
                        subtract!(out_i, increments[(i,j,l)])
                        subtract!(out_i, increments[(i,k,l)])
                        subtract!(out_i, increments[(j,k,l)])
                        subtract!(out_i, increments[(i,j)])
                        subtract!(out_i, increments[(i,k)])
                        subtract!(out_i, increments[(i,l)])
                        subtract!(out_i, increments[(j,k)])
                        subtract!(out_i, increments[(j,l)])
                        subtract!(out_i, increments[(k,l)])
                        subtract!(out_i, increments[(i,)])
                        subtract!(out_i, increments[(j,)])
                        subtract!(out_i, increments[(k,)])
                        subtract!(out_i, increments[(l,)])
                        
                        increments[(i,j,k,l)] = out_i
                        verbose < 2 || display(out_i) 
                    end
                end
            end
        end
    end

    # 5-body
    if nbody >= 5 
        for i in 1:n_clusters
            for j in i+1:n_clusters
                for k in j+1:n_clusters
                    for l in k+1:n_clusters
                        for m in l+1:n_clusters
                        
                            #screen((i, j, k, l), thresh, increments) == false || continue
                        
                            ci = clusters[i]
                            cj = clusters[j]
                            ck = clusters[k]
                            cl = clusters[l]
                            cm = clusters[m]
                            out_i = compute_increment(ints, [ci, cj, ck, cl, cm], fspace, rdm1a, rdm1b, verbose=verbose)

                            # subtract the hartree fock data so that we have an increment from HF
                            subtract!(out_i, ref_data)

                            # subtract the lower orders 
                            subtract!(out_i, increments[(i,j,k,l)])
                            subtract!(out_i, increments[(i,j,k,m)])
                            subtract!(out_i, increments[(i,j,l,m)])
                            subtract!(out_i, increments[(i,k,l,m)])
                            subtract!(out_i, increments[(j,k,l,m)])

                            subtract!(out_i, increments[(i,j,k)])
                            subtract!(out_i, increments[(i,j,l)])
                            subtract!(out_i, increments[(i,j,m)])
                            subtract!(out_i, increments[(i,k,l)])
                            subtract!(out_i, increments[(i,k,m)])
                            subtract!(out_i, increments[(i,l,m)])
                            subtract!(out_i, increments[(j,k,l)])
                            subtract!(out_i, increments[(j,k,m)])
                            subtract!(out_i, increments[(j,l,m)])
                            subtract!(out_i, increments[(k,l,m)])

                            subtract!(out_i, increments[(i,j)])
                            subtract!(out_i, increments[(i,k)])
                            subtract!(out_i, increments[(i,l)])
                            subtract!(out_i, increments[(i,m)])
                            subtract!(out_i, increments[(j,k)])
                            subtract!(out_i, increments[(j,l)])
                            subtract!(out_i, increments[(j,m)])
                            subtract!(out_i, increments[(k,l)])
                            subtract!(out_i, increments[(k,m)])
                            subtract!(out_i, increments[(l,m)])
                            subtract!(out_i, increments[(i,j)])
                            
                            subtract!(out_i, increments[(i,)])
                            subtract!(out_i, increments[(j,)])
                            subtract!(out_i, increments[(k,)])
                            subtract!(out_i, increments[(l,)])
                            subtract!(out_i, increments[(m,)])

                            increments[(i,j,k,l,m)] = out_i
                            verbose < 2 || display(out_i) 
                        end
                    end
                end
            end
        end
    end

    @printf(" Number of increments: %i\n", length(increments))
    @printf(" Add results\n")
    final_data = deepcopy(ref_data) 
    for (key,inc) in increments
        #display(inc)
        add!(final_data, inc)
    end
    

    Enew = compute_energy(ints, final_data)
    @printf(" Summed quantities\n")
    @printf("   E0:      %12.8f\n", E0)
    @printf("   tr(HG):  %12.8f\n", Enew)
    println(" Input data:")
    display(ref_data)
    println(" Output data:")
    display(final_data)
         
    if false 
        ci = Cluster(1,[1,2,3,4]) 
        cj = Cluster(2,[5,6,7,8]) 
        out_i = compute_increment(ints, [ci, cj], [(4,4),(0,0)], rdm1a, rdm1b)

        display(out_i) 
    end
    return final_data, increments
end



