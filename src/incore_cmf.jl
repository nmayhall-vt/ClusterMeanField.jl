using ClusterMeanField





"""
    QCBase.compute_energy(ints::InCoreInts{T}, rdm1s::Dict{Integer,RDM1{T}}, rdm2s::Dict{Integer,RDM2{T}}, clusters::Vector{MOCluster}; verbose=0) where T

Compute the energy of a cluster-wise product state (CMF),
specified by a list of 1 and 2 particle rdms local to each cluster.
This method uses the full system integrals.

- `ints::InCoreInts`: integrals for full system
- `rdm1s`: dictionary (`ci.idx => RDM1`) of 1rdms from each cluster
- `rdm2s`: dictionary (`ci.idx => RDM2`) of 2rdms from each cluster
- `clusters::Vector{MOCluster}`: vector of cluster objects

return the total CMF energy
"""
function QCBase.compute_energy(ints::InCoreInts{T}, rdm1s::Dict{Integer,RDM1{T}}, rdm2s::Dict{Integer,RDM2{T}}, clusters::Vector{MOCluster}; verbose=0) where T
    e1 = zeros((length(clusters),1))
    e2 = zeros((length(clusters),length(clusters)))
    for ci in clusters
        noi = n_orb(ints)
        ints_i = subset(ints, ci.orb_list)

        e1[ci.idx] = compute_energy(ints_i, rdm1s[ci.idx], rdm2s[ci.idx])
    end
    for ci in clusters
        for cj in clusters
            if ci.idx >= cj.idx
                continue
            end
            v_pqrs = ints.h2[ci.orb_list, ci.orb_list, cj.orb_list, cj.orb_list]
            v_psrq = ints.h2[ci.orb_list, cj.orb_list, cj.orb_list, ci.orb_list]
            tmp = 0


            @tensor begin
                tmp  = v_pqrs[p,q,r,s] * rdm1s[ci.idx].a[p,q] * rdm1s[cj.idx].a[r,s]
                tmp -= v_psrq[p,s,r,q] * rdm1s[ci.idx].a[p,q] * rdm1s[cj.idx].a[r,s]

                tmp += v_pqrs[p,q,r,s] * rdm1s[ci.idx].b[p,q] * rdm1s[cj.idx].b[r,s]
                tmp -= v_psrq[p,s,r,q] * rdm1s[ci.idx].b[p,q] * rdm1s[cj.idx].b[r,s]

                tmp += v_pqrs[p,q,r,s] * rdm1s[ci.idx].a[p,q] * rdm1s[cj.idx].b[r,s]

                tmp += v_pqrs[p,q,r,s] * rdm1s[ci.idx].b[p,q] * rdm1s[cj.idx].a[r,s]
            end


            e2[ci.idx, cj.idx] = tmp
        end
    end
    if verbose>1
        for ei = 1:length(e1)
            @printf(" MOCluster %3i E =%12.8f\n",ei,e1[ei])
        end
    end
    return ints.h0 + sum(e1) + sum(e2)
end


"""
    cmf_ci_iteration(ints::InCoreInts, clusters::Vector{MOCluster}, rdm1a, rdm1b, fspace; verbose=1)

Perform single CMF-CI iteration, returning new energy, and density
"""
function cmf_ci_iteration(ints::InCoreInts{T}, clusters::Vector{MOCluster}, in_rdm1::RDM1{T}, fspace; verbose=1,sequential=false) where T
    rdm1 = deepcopy(in_rdm1)
    rdm1_dict = Dict{Integer,RDM1{T}}()
    rdm2_dict = Dict{Integer,RDM2{T}}()
    
    for ci in clusters
        flush(stdout)

        ansatz = FCIAnsatz(length(ci), fspace[ci.idx][1],fspace[ci.idx][2])
        verbose < 2 || display(ansatz)
        ints_i = subset(ints, ci, rdm1)
        
        d1a = rdm1.a[ci.orb_list, ci.orb_list]
        d1b = rdm1.b[ci.orb_list, ci.orb_list]

        na = fspace[ci.idx][1]
        nb = fspace[ci.idx][2]

        no = length(ci)

        d1 = RDM1(no)
        d2 = RDM2(no)

        e = 0.0
        if ansatz.dim == 1
       
            da = zeros(T, no, no)
            db = zeros(T, no, no)
            if ansatz.na == no
                da = Matrix(1.0I, no, no)
            end
            if ansatz.nb == no
                db = Matrix(1.0I, no, no)
            end

            d1 = RDM1(da,db)
            d2 = RDM2(d1)
            
            e = compute_energy(ints_i, d1)
            verbose < 2 || @printf(" Slater Det Energy: %12.8f\n", e)
        else
            #
            # run PYSCF FCI
            #e, d1a,d1b, d2 = pyscf_fci(ints_i,fspace[ci.idx][1],fspace[ci.idx][2], verbose=verbose)
           
            solver = SolverSettings(verbose=1)
            solution = solve(ints_i, ansatz, solver)
            d1a, d1b, d2aa, d2bb, d2ab = compute_1rdm_2rdm(solution)
            
            d1 = RDM1(d1a, d1b)
            d2 = RDM2(d2aa, d2ab, d2bb)

#            pyscf = pyimport("pyscf")
#            fci = pyimport("pyscf.fci")
#            cisolver = pyscf.fci.direct_spin1.FCI()
#            cisolver.max_cycle = 100 
#            cisolver.conv_tol = 1e-8
#            nelec = na + nb
#            e, vfci = cisolver.kernel(ints_i.h1, ints_i.h2, no, (na,nb), ecore=ints_i.h0)
#            (d1a, d1b), (d2aa, d2ab, d2bb)  = cisolver.make_rdm12s(vfci, no, (na,nb))
#
#            d1 = RDM1(d1a, d1b)
#            d2 = RDM2(d2aa, d2ab, d2bb)

        end

        rdm1_dict[ci.idx] = d1
        rdm2_dict[ci.idx] = d2

        if sequential==true
            rdm1.a[ci.orb_list,ci.orb_list] = d1.a
            rdm1.b[ci.orb_list,ci.orb_list] = d1.b
        end
    end
    e_curr = compute_energy(ints, rdm1_dict, rdm2_dict, clusters, verbose=verbose)
    
    if verbose > 1
        @printf(" CMF-CI Curr: Elec %12.8f Total %12.8f\n", e_curr-ints.h0, e_curr)
    end

    return e_curr, rdm1_dict, rdm2_dict
end



"""
    cmf_ci(ints, clusters, fspace, in_rdm1a, in_rdm1b; 
                max_iter=10, dconv=1e-6, econv=1e-10, verbose=1,sequential=false)

Optimize the 1RDM for CMF-CI

#Arguments
- `ints::InCoreInts`: integrals for full system
- `clusters::Vector{MOCluster}`: vector of cluster objects
- `fspace::Vector{Vector{Integer}}`: vector of particle number occupations for each cluster specifying the sectors of fock space 
- `in_rdm1a`: initial guess for 1particle density matrix for alpha electrons
- `in_rdm1b`: initial guess for 1particle density matrix for beta electrons
- `dconv`: Convergence threshold for change in density 
- `econv`: Convergence threshold for change in energy 
- `sequential`: Use the density matrix of the previous cluster in a cMF iteration to form effective integrals. Improves comvergence, may depend on cluster orderings   
- `verbose`: Printing level 
"""
function cmf_ci(ints, clusters, fspace, in_rdm1::RDM1; 
                max_iter=10, dconv=1e-6, econv=1e-10, verbose=1,sequential=false)
    rdm1 = deepcopy(in_rdm1)
    energies = []
    e_prev = 0

    rdm1_dict = 0
    rdm2_dict = 0
    rdm1_dict = Dict{Integer,Array}()
    rdm2_dict = Dict{Integer,Array}()
    # rdm2_dict = Dict{Integer, Array}()
    for iter = 1:max_iter
        if verbose > 1
            println()
            println(" ------------------------------------------ ")
            println(" CMF CI Iter: ", iter)
            println(" ------------------------------------------ ")
        end
        e_curr, rdm1_dict, rdm2_dict = cmf_ci_iteration(ints, clusters, rdm1, fspace, verbose=verbose,sequential=sequential)
        rdm1_curr = assemble_full_rdm(clusters, rdm1_dict)

        append!(energies,e_curr)
        error = (rdm1_curr.a+rdm1_curr.b) - (rdm1.a+rdm1.b)
        d_err = norm(error)
        e_err = e_curr-e_prev
        if verbose>1
            @printf(" CMF-CI Energy: %12.8f | Change: RDM: %6.1e Energy %6.1e\n\n", e_curr, d_err, e_err)
        end
        e_prev = e_curr*1
        rdm1 = rdm1_curr
        if (abs(d_err) < dconv) && (abs(e_err) < econv)
            if verbose>1
                @printf("*CMF-CI: Elec %12.8f Total %12.8f\n", e_curr-ints.h0, e_curr)
            end
            break
        end
    end
    if verbose>0
        println(" Energy per Iteration:")
        for i in energies
            @printf(" Elec: %12.8f Total: %12.8f\n", i-ints.h0, i)
        end
    end
    return e_prev, rdm1_dict, rdm2_dict
end










"""
    cmf_oo(ints::InCoreInts, clusters::Vector{MOCluster}, fspace, dguess_a, dguess_b; 
                max_iter_oo=100, max_iter_ci=100, gconv=1e-6, verbose=0, method="bfgs", alpha=nothing,sequential=false)

Do CMF with orbital optimization

#Arguments
- `ints::InCoreInts`: integrals for full system
- `clusters::Vector{MOCluster}`: vector of cluster objects
- `fspace::Vector{Vector{Integer}}`: vector of particle number occupations for each cluster specifying the sectors of fock space 
- `dguess_a`: initial guess for 1particle density matrix
- `max_iter_oo`: Max iter for the orbital optimization iterations 
- `max_iter_ci`: Max iter for the cmf iteration for the cluster states 
- `gconv`: Convergence threshold for change in gradient of energy 
- `sequential`: If true use the density matrix of the previous cluster in a cMF iteration to form effective integrals. Improves comvergence, may depend on cluster orderings   
- `verbose`: Printing level 
- `method`: optimization method
"""
function cmf_oo(ints::InCoreInts, clusters::Vector{MOCluster}, fspace, dguess::RDM1; 
                max_iter_oo=100, 
                max_iter_ci=100, 
                gconv=1e-6, 
                verbose=0, 
                method="bfgs", 
                alpha=nothing,
                sequential=false)
    norb = size(ints.h1)[1]
    #kappa = zeros(norb*(norb-1))
    # e, da, db = cmf_oo_iteration(ints, clusters, fspace, max_iter_ci, dguess, kappa)

    function g_numerical(k)
        stepsize = 1e-6
        grad = zeros(size(k))
        for (ii,i) in enumerate(k)
            k1 = deepcopy(k)
            k1[ii] += stepsize
            e1 = f(k1) 
            k2 = deepcopy(k)
            k2[ii] -= stepsize
            e2 = f(k2) 
            grad[ii] = (e1-e2)/(2*stepsize)
        end
        g_curr = norm(grad)
        return grad
    end

    #   
    #   Initialize optimization data
    #
    e_prev = 0
    e_curr = 0
    g_curr = 0
    e_err = 0
    #da = zeros(size(ints.h1))
    #db = zeros(size(ints.h1))
    d = deepcopy(dguess)
    d1 = deepcopy(dguess)
    d2 = deepcopy(dguess)
    
    da = deepcopy(dguess.a)
    db = deepcopy(dguess.b)

    da1 = deepcopy(dguess.a)
    db1 = deepcopy(dguess.b)

    da2 = deepcopy(dguess.a)
    db2 = deepcopy(dguess.b)

    iter = 0
    kappa = zeros(norb*(norb-1)÷2)

    #
    #   Define Objective function (energy)
    #
    function f(k)
        #display(norm(k))
        K = unpack_gradient(k, norb)
        U = exp(K)
        ints2 = orbital_rotation(ints,U)
        d1  = orbital_rotation(d,U)
        e, rdm1_dict, rdm2_dict = cmf_ci(ints2, clusters, fspace, d1, 
                                         dconv=gconv/10.0, verbose=0,sequential=sequential)
        d2  = orbital_rotation(d1,U')
        e_err = e-e_curr
        e_curr = e
        return e
    end

    #   
    #   Define Callback for logging and checking for convergence
    #
    function callback(k)
       
        # reset initial RDM guess for each cmf_ci
        da = deepcopy(da2)
        db = deepcopy(db2)

        #if e_err > 0
        #    @warn " energy increased"
        #    return true
        #end
        iter += 1
        if (g_curr < gconv) 
            @printf("*ooCMF Iter: %4i Total= %16.12f Active= %16.12f G= %12.2e\n", iter, e_curr, e_curr-ints.h0, g_curr)
            return true 
        else
            @printf(" ooCMF Iter: %4i Total= %16.12f Active= %16.12f G= %12.2e\n", iter, e_curr, e_curr-ints.h0, g_curr)
            return false 
        end
    end

    function g2(kappa)
        norb = n_orb(ints)
        # println(" In g_analytic")
        K = unpack_gradient(kappa, norb)
        U = exp(K)
        ints2 = orbital_rotation(ints,U)
        d1 = orbital_rotation(d,U)
        
        e, rdm1_dict, rdm2_dict = cmf_ci(ints2, clusters, fspace, d1, 
                                         dconv=gconv/10.0, verbose=verbose)

        gd1, gd2 = assemble_full_rdm(clusters, rdm1_dict, rdm2_dict)
        gout = build_orbital_gradient(ints2, gd1, gd2)
        g_curr = norm(gout)
        return gout
    end
    #
    #   Define Gradient function
    #
    function g(kappa)
        norb = size(ints.h1)[1]
        # println(" In g_analytic")
        K = unpack_gradient(kappa, norb)
        U = exp(K)
        #println(size(U), size(kappa))
        ints2 = orbital_rotation(ints,U)
        d1 = orbital_rotation(d,U)
        
        e, rdm1_dict, rdm2_dict = cmf_ci(ints2, clusters, fspace, d1, 
                                         dconv=gconv/10.0, verbose=verbose)
        grad = zeros(size(ints2.h1))
        for ci in clusters
            grad_1 = grad[:,ci.orb_list]
            h_1	   = ints2.h1[:,ci.orb_list]
            v_111  = ints2.h2[:, ci.orb_list, ci.orb_list, ci.orb_list]
            @tensor begin
                grad_1[p,q] += v_111[p,v,u,w] * rdm2_dict[ci.idx].aa[q,v,u,w]
                grad_1[p,q] += v_111[p,v,u,w] * rdm2_dict[ci.idx].bb[q,v,u,w]
                grad_1[p,q] += v_111[p,v,u,w] * rdm2_dict[ci.idx].ab[q,v,u,w]
                grad_1[p,q] += v_111[p,v,u,w] * rdm2_dict[ci.idx].ab[u,w,q,v]
                grad_1[p,q] += h_1[p,r] * (rdm1_dict[ci.idx].a[r,q]+rdm1_dict[ci.idx].b[r,q])
            end
            for cj in clusters
                if ci.idx == cj.idx
                    continue
                end
                v_212 = ints2.h2[:,cj.orb_list, ci.orb_list, cj.orb_list]
                v_122 = ints2.h2[:,ci.orb_list, cj.orb_list, cj.orb_list]
                d1 = rdm1_dict[ci.idx].a + rdm1_dict[ci.idx].b
                d2 = rdm1_dict[cj.idx].a + rdm1_dict[cj.idx].b
                
                d1a = rdm1_dict[ci.idx].a
                d1b = rdm1_dict[ci.idx].b
                d2a = rdm1_dict[cj.idx].a
                d2b = rdm1_dict[cj.idx].b

                @tensor begin
                    #grad_1[p,q] += v_122[p,v,u,w] * d1[q,v] * d2[w,u]
                    #grad_1[p,q] -= .5*v_212[p,v,u,w] * d1[q,u] * d2[w,v]
                    grad_1[p,q] += v_122[p,v,u,w] * d1[q,v] * d2[w,u]
                    grad_1[p,q] -= v_212[p,v,u,w] * d1a[q,u] * d2a[w,v]
                    grad_1[p,q] -= v_212[p,v,u,w] * d1b[q,u] * d2b[w,v]
                end
            end
            grad[:,ci.orb_list] .= -2*grad_1
        end
        grad = grad'-grad
        gout = pack_gradient(grad, norb)
        g_curr = norm(gout)
        return gout
    end

    if false
        grad1 = g_numerical(kappa)
        grad2 = g(kappa)
        grad3 = g2(kappa)
        display(round.(unpack_gradient(grad1, norb),digits=6))
        display(round.(unpack_gradient(grad2, norb),digits=6))
        display(round.(unpack_gradient(grad3, norb),digits=6))
        error("nick") 
    end
    #display("here:")
    #gerr = g_numerical(kappa) - g(kappa)
    #display(norm(gerr))
    #for i in gerr
    #    @printf(" err: %12.8f\n",i)
    #end
    #return

    if (method=="bfgs") || (method=="cg") || (method=="gd")
        optmethod = BFGS()
        if method=="cg"
            optmethod = ConjugateGradient()
        elseif method=="gd"

            if alpha == nothing
                optmethod = GradientDescent()
            else 
                optmethod = GradientDescent(alphaguess=alpha)
            end
        end

        options = Optim.Options(
                                callback = callback, 
                                g_tol=gconv,
                                iterations=max_iter_oo,
                               )

        #res = optimize(f, g_numerical, kappa, optmethod, options; inplace = false )
        res = optimize(f, g, kappa, optmethod, options; inplace = false )
        #res = optimize(f, g2, kappa, optmethod, options; inplace = false )
        summary(res)
        e = Optim.minimum(res)
        display(res)
        @printf("*ooCMF %12.8f \n", e)

        kappa = Optim.minimizer(res)
        K = unpack_gradient(kappa, norb)
        U = exp(K)
        da1 = U'*da*U
        db1 = U'*db*U
        return e, U, da1, db1
    elseif method=="diis"
        res = do_diis(f, g, callback, kappa, gconv, max_iter_oo, method)
    end

end



function do_diis(f,g,callback,kappa, gconv,max_iter, method)
    throw("Not yet implemented")
end


"""
    unpack_gradient(kappa,norb)
"""
function unpack_gradient(kappa,norb)
    length(kappa) == norb*(norb-1)÷2 || throw(DimensionMismatch)
    K = zeros(norb,norb)
    ind = 1
    for i in 1:norb
        for j in i+1:norb
            K[i,j] = kappa[ind]
            K[j,i] = -kappa[ind]
            ind += 1
        end
    end
    return K
end
"""
    pack_gradient(K,norb)
"""
function pack_gradient(K,norb)
    length(K) == norb*norb || throw(DimensionMismatch)
    kout = zeros(norb*(norb-1)÷2)
    ind = 1
    for i in 1:norb
        for j in i+1:norb
            kout[ind] = K[i,j]
            ind += 1
        end
    end
    return kout
end



"""
    assemble_full_rdm(clusters::Vector{MOCluster}, rdm1s::Dict{Integer, RDM1{T}}) where T

Return spin summed 1 and 2 RDMs
"""
function assemble_full_rdm(clusters::Vector{MOCluster}, rdm1s::Dict{Integer, RDM1{T}}) where T
    norb = sum([length(i) for i in clusters])
    d1 = RDM1(norb)

    for ci in clusters
        d1.a[ci.orb_list, ci.orb_list] .= rdm1s[ci.idx].a
        d1.b[ci.orb_list, ci.orb_list] .= rdm1s[ci.idx].b
    end
    
    return d1
end

"""
    assemble_full_rdm(clusters::Vector{MOCluster}, rdm1s::Dict{Integer, Array}, rdm2s::Dict{Integer, Array})

Return full system 1 and 2 RDMs
"""
function assemble_full_rdm(clusters::Vector{MOCluster}, rdm1s::Dict{Integer, RDM1{T}}, rdm2s::Dict{Integer, RDM2{T}}) where T
    norb = sum([length(i) for i in clusters])

    rdm1 = RDM1(norb)
    rdm2 = RDM2(norb)
    
    for ci in clusters
        rdm1.a[ci.orb_list, ci.orb_list] .= rdm1s[ci.idx].a
        rdm1.b[ci.orb_list, ci.orb_list] .= rdm1s[ci.idx].b
    end
   
    rdm2 = RDM2(rdm1)
    for ci in clusters
        rdm2.aa[ci.orb_list, ci.orb_list, ci.orb_list, ci.orb_list] .= rdm2s[ci.idx].aa
        rdm2.ab[ci.orb_list, ci.orb_list, ci.orb_list, ci.orb_list] .= rdm2s[ci.idx].ab
        rdm2.bb[ci.orb_list, ci.orb_list, ci.orb_list, ci.orb_list] .= rdm2s[ci.idx].bb
    end
    return rdm1, rdm2
end


#
"""
    orbital_gradient_analytical(ints, clusters, kappa, fspace, da, db;
                                    gconv = 1e-8,
                                    verbose = 1)
Compute orbital gradient analytically
```math
w_{pq} = 2(f_{pq} - f_{qp})
```
```math
f_{pq} = h_{pr}P_{rq} + 2 <rs||tp> G_{pstq}
```
"""
function orbital_gradient_analytical(ints, clusters, kappa, fspace, rdm::RDM1;
                                    ci_conv = 1e-8,
                                    verbose = 0)
#={{{=#
    norb = n_orb(ints)
    # println(" In g_analytic")
    K = unpack_gradient(kappa, norb)
    U = exp(K)
    #display("nick U")
    #display(U)
    ints2 = orbital_rotation(ints,U)
    d1 = orbital_rotation(rdm,U)

    e, rdm1_dict, rdm2_dict = cmf_ci(ints2, clusters, fspace, d1, dconv=ci_conv, verbose=verbose)
    grad = zeros(size(ints2.h1))
    for ci in clusters
        grad_1 = grad[:,ci.orb_list]
        h_1	   = ints2.h1[:,ci.orb_list]
        v_111  = ints2.h2[:, ci.orb_list, ci.orb_list, ci.orb_list]
        @tensor begin
            grad_1[p,q] += v_111[p,v,u,w] * rdm2_dict[ci.idx].aa[q,v,u,w]
            grad_1[p,q] += v_111[p,v,u,w] * rdm2_dict[ci.idx].bb[q,v,u,w]
            grad_1[p,q] += v_111[p,v,u,w] * rdm2_dict[ci.idx].ab[q,v,u,w]
            grad_1[p,q] += v_111[p,v,u,w] * rdm2_dict[ci.idx].ab[u,v,q,w]
            grad_1[p,q] += h_1[p,r] * (rdm1_dict[ci.idx].a[r,q]+rdm1_dict[ci.idx].b[r,q])
        end
        for cj in clusters
            if ci.idx == cj.idx
                continue
            end
            v_212 = ints2.h2[:,cj.orb_list, ci.orb_list, cj.orb_list]
            v_122 = ints2.h2[:,ci.orb_list, cj.orb_list, cj.orb_list]
            d1 = rdm1_dict[ci.idx].a + rdm1_dict[ci.idx].b
            d2 = rdm1_dict[cj.idx].a + rdm1_dict[cj.idx].b

            d1a = rdm1_dict[ci.idx].a
            d1b = rdm1_dict[ci.idx].b
            d2a = rdm1_dict[cj.idx].a
            d2b = rdm1_dict[cj.idx].b

            @tensor begin
                #grad_1[p,q] += v_122[p,v,u,w] * d1[q,v] * d2[w,u]
                #grad_1[p,q] -= .5*v_212[p,v,u,w] * d1[q,u] * d2[w,v]
                grad_1[p,q] += v_122[p,v,u,w] * d1[q,v] * d2[w,u]
                grad_1[p,q] -= v_212[p,v,u,w] * d1a[q,u] * d2a[w,v]
                grad_1[p,q] -= v_212[p,v,u,w] * d1b[q,u] * d2b[w,v]
            end
        end
        grad[:,ci.orb_list] .= -2*grad_1
    end
    grad .= grad .- grad'
    #grad = U*grad*U'
    gout = pack_gradient(grad, norb)
    return gout
end
#=}}}=#

"""
    orbital_objective_function(ints, clusters, kappa, fspace, da, db; 
                                    ci_conv     = 1e-9,
                                    sequential  = false,
                                    verbose     = 1)
Objective function to minimize in OO-CMF
"""
function orbital_objective_function(ints, clusters, kappa, fspace, rdm::RDM1; 
                                    ci_conv     = 1e-9,
                                    sequential  = false,
                                    verbose     = 0)

    #davg = .5*(da + db)
    norb = n_orb(ints)
    K = unpack_gradient(kappa, norb)
    #U = Matrix(1.0I,norb,norb)
    U = exp(K)
    #display("nick U")
    #display(U)
    #display("nick K")
    #display(K)
    ints2 = orbital_rotation(ints,U)
    d1 = orbital_rotation(rdm,U)
    e, rdm1_dict, rdm2_dict = cmf_ci(ints2, clusters, fspace, d1, 
        dconv=ci_conv, 
        verbose=verbose,
        sequential=sequential)
    return e
end

"""
    orbital_gradient_numerical(ints, clusters, kappa, fspace, da, db; 
                                    gconv = 1e-8, 
                                    verbose = 1,
                                    stepsize = 1e-6)
Compute orbital gradient with finite difference
"""
function orbital_gradient_numerical(ints, clusters, kappa, fspace, d::RDM1; 
                                    ci_conv = 1e-10, 
                                    verbose = 0,
                                    stepsize = 1e-6)
    grad = zeros(size(kappa))
    for (ii,i) in enumerate(kappa)
        
        #ii == 2 || continue
    
        k1 = deepcopy(kappa)
        k1[ii] += stepsize
        e1 = orbital_objective_function(ints, clusters, k1, fspace, d, ci_conv=ci_conv, verbose=verbose) 
        
        k2 = deepcopy(kappa)
        k2[ii] -= stepsize
        e2 = orbital_objective_function(ints, clusters, k2, fspace, d, ci_conv=ci_conv, verbose=verbose) 
        
        grad[ii] = (e1-e2)/(2*stepsize)
        #println(e1)
    end
    return grad
end


