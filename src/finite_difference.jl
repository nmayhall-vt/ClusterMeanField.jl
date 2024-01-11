using ClusterMeanField
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

    norb = n_orb(ints)
    K = unpack_gradient(kappa, norb)
    U = exp(K)
    ints2 = orbital_rotation(ints,U)
    d1 = orbital_rotation(rdm,U)
    e, rdm1_dict, rdm2_dict = cmf_ci(ints2, clusters, fspace, d1, tol_ci=ci_conv, verbose=verbose,sequential=sequential)
    return e
end
"""
    orbital_objective_function(ints, clusters, kappa, fspace, ansatze::Vector{Ansatz}, da, db; 
                                    ci_conv     = 1e-9,
                                    sequential  = false,
                                    verbose     = 1)
Objective function to minimize in OO-CMF
"""
function orbital_objective_function(ints, clusters, kappa, fspace,ansatze::Vector{<:Ansatz}, rdm::RDM1{T}; 
                                    ci_conv     = 1e-9,
                                    sequential  = false,
                                    verbose     = 0) where T

    norb = n_orb(ints)
    K = unpack_gradient(kappa, norb)
    U = exp(K)
    ints_tmp = orbital_rotation(ints,U)
    e, rdm1_dict, _ = cmf_ci(ints_tmp, clusters, fspace, ansatze, orbital_rotation(rdm, U), verbose=verbose)
    return e
end

"""
    orbital_gradient_numerical(ints, clusters, kappa, fspace, ansatze::Vector{Ansatz}, da, db; 
                                    gconv = 1e-8, 
                                    verbose = 1,
                                    stepsize = 1e-6)
Compute orbital gradient with finite difference
"""
function orbital_gradient_numerical(ints, clusters, kappa, fspace, ansatze::Vector{<:Ansatz}, d::RDM1; 
                                    ci_conv = 1e-10, 
                                    verbose = 0,
                                    stepsize = 1e-6)
    grad = zeros(size(kappa))
    for (ii,i) in enumerate(kappa)
        
        #ii == 2 || continue
    
        k1 = deepcopy(kappa)
        k1[ii] += stepsize
        e1 = orbital_objective_function(ints, clusters, k1, fspace, ansatze, d, ci_conv=ci_conv, verbose=verbose) 
        
        k2 = deepcopy(kappa)
        k2[ii] -= stepsize
        e2 = orbital_objective_function(ints, clusters, k2, fspace, ansatze, d, ci_conv=ci_conv, verbose=verbose) 
        
        grad[ii] = (e1-e2)/(2*stepsize)
        #println(e1)
    end
    return grad
end

"""
    orbital_gradient_numerical(ints, clusters, kappa, fspace, da, db; 
                                    ci_conv = 1e-8, 
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
    orbital_hessian_finite_difference(ints, clusters, kappa, fspace, d; 
                                    ci_conv = 1e-8, 
                                    verbose = 1,
                                    stepsize = 1e-6)
Compute orbital hessian with finite difference for cMF using orbital energy
"""
function orbital_hessian_finite_difference(ints, clusters, kappa, fspace, d::RDM1; ci_conv = 1e-10, verbose = 0,stepsize = 1e-5)
    n = length(kappa)
    # error("here")
    hessian = zeros(n, n)
    for i in 1:n
        kappa_plus=deepcopy(kappa)
        kappa_plus[i]+=stepsize

        kappa_minus=deepcopy(kappa)
        kappa_minus[i]-=stepsize
        #hessian[i,i]=(orbital_objective_function(ints, clusters, kappa_plus, fspace, d, ci_conv = ci_conv, verbose = verbose)-2*orbital_objective_function(ints, clusters, kappa, fspace, d, ci_conv = ci_conv, verbose = verbose)+orbital_objective_function(ints, clusters, kappa_minus, fspace, d, ci_conv = ci_conv, verbose = verbose))/(stepsize^2)
        kappa_2plus=deepcopy(kappa)
        kappa_2plus[i] +=(2*stepsize)
        kappa_2minus=deepcopy(kappa)
        kappa_2minus[i]-=(2*stepsize)


        k0 = deepcopy(kappa)
        e0 = orbital_objective_function(ints, clusters, k0, fspace, d, ci_conv=ci_conv, verbose=verbose) 

        k1 = deepcopy(kappa)
        k1[i] += stepsize
        e1 = orbital_objective_function(ints, clusters, k1, fspace, d, ci_conv=ci_conv, verbose=verbose) 
        
        k2 = deepcopy(kappa)
        k2[i] -= stepsize
        e2 = orbital_objective_function(ints, clusters, k2, fspace, d, ci_conv=ci_conv, verbose=verbose) 
        
        grad = (e1-e2)/(2*stepsize)
        hessian[i,i] = (e1 - 2*e0 + e2) / (stepsize^2)

        @printf(" e0: %12.8f e1: %12.8f e2: %12.8f grad: %12.8f\n", e0, e1, e2, grad)
        
        for j in (i+1):n
            # Perturb parameters in both directions
            kappa_plus_plus = deepcopy(kappa)
            kappa_plus_plus[i] += stepsize
            kappa_plus_plus[j] += stepsize

            kappa_plus_minus = deepcopy(kappa)
            kappa_plus_minus[i] += stepsize
            kappa_plus_minus[j] -= stepsize

            kappa_minus_plus = deepcopy(kappa)
            kappa_minus_plus[i] -= stepsize
            kappa_minus_plus[j] += stepsize

            kappa_minus_minus = deepcopy(kappa)
            kappa_minus_minus[i] -= stepsize
            kappa_minus_minus[j] -= stepsize

            # Calculate finite difference approximation   
            hessian[i, j] = (orbital_objective_function(ints, clusters, kappa_plus_plus, fspace, d, ci_conv = ci_conv, verbose = verbose) -
                            orbital_objective_function(ints, clusters, kappa_plus_minus, fspace, d, ci_conv = ci_conv, verbose = verbose) -
                            orbital_objective_function(ints, clusters, kappa_minus_plus, fspace, d, ci_conv = ci_conv, verbose = verbose) +
                            orbital_objective_function(ints, clusters, kappa_minus_minus, fspace, d, ci_conv = ci_conv, verbose = verbose))/(4*(stepsize^2))
            # Fill in the symmetric entry
            hessian[j, i] = hessian[i, j]
        end
    end

    return hessian
end


"""
    orbital_hessian_numerical(ints, clusters, kappa, fspace, d; 
                                    ci_conv = 1e-8, 
                                    verbose = 1,
                                    stepsize = 1e-6)
Compute orbital hessian with finite difference for cMF using orbital gradient 
"""
function orbital_hessian_numerical(ints, clusters, kappa, fspace, d::RDM1; ci_conv = 1e-10, verbose = 0,step_size = 1e-5,zero_intra_rots = true,maxiter_ci = 100, maxiter_d1 = 100, tol_oo = 1e-6, tol_d1 = 1e-7, tol_ci = 1e-8, alpha  = .1,sequential = false)
    n = length(kappa)
    #error("here.....")
    hessian = zeros(n, n)
    function step_numerical!(ints, d1, k)
        norb = n_orb(ints)
        K = unpack_gradient(k, norb)
        if zero_intra_rots
            # Remove intracluster rotations
            for ci in clusters
                K[ci.orb_list, ci.orb_list] .= 0
            end
        end
       
        Ui = exp(K)
        
        tmp_ints = orbital_rotation(ints,Ui)

        e, rdm1_dict, rdm2_dict = cmf_ci(tmp_ints, clusters, fspace, d1, 
                                         maxiter_d1 = maxiter_d1, 
                                         maxiter_ci = maxiter_ci, 
                                         tol_d1     = tol_d1, 
                                         tol_ci     = tol_ci, 
                                         verbose    = 0, 
                                         sequential = sequential)
        
        gd1, gd2 = assemble_full_rdm(clusters, rdm1_dict, rdm2_dict)
        G = build_orbital_gradient(tmp_ints, gd1, gd2)
        # G=orbital_gradient_numerical(ints,clusters,k,fspace,gd1)
        return  G
    end
    
    norb = n_orb(ints)
    #central difference 
    for i in 1:n
        x_plus_i = deepcopy(kappa)
        x_minus_i = deepcopy(kappa)
        x_plus_i[i] += step_size
        x_minus_i[i] -= step_size
        g_plus_i=step_numerical!(ints,d,x_plus_i)
        g_minus_i=step_numerical!(ints,d,x_minus_i)
        
        gnum = (g_plus_i .- g_minus_i)./(2*step_size)
        display(gnum)
        for j in 1:n
            hessian[i,j] = gnum[j]
            #error("dfdg")
        end

        
    end
    # display(hessian)
    return hessian
end
"""create_interaction_vector(clusters,total_elements)
        compute the projection vector for redundant rotations
"""


function create_interaction_vector(clusters, total_elements)
    blocks = [collect(t[1]:t[end]) for t in clusters]
    matrix=ones(total_elements,total_elements)
    kout = zeros(total_elements*(total_elements-1)÷2)
    ind = 1
    for cluster in blocks
        for i in 1:total_elements
            for j in 1:total_elements
                if i in cluster && j in cluster 
                    matrix[i,j]=0
                end
            end
        end
    end
            
    for i in 1:total_elements
        for j in i+1:total_elements
            kout[ind] = matrix[i,j]
            ind += 1
        end
    end           
    return kout       
end
"""get_global_pair(local_pair_vector)
        computes the global  pairs for which energy is invariant to orbital rotation
"""
function get_global_pair(local_pair_vector)
    global_pair_vector =[]
    offset = 0
    # Loop through each sub-vector in the original vector
    for sub_vector in local_pair_vector
        # Calculate the offset based on the length of the sub-vector
        desired_sub_vector= [(x + offset , y + offset) for (i, (x, y)) in enumerate(sub_vector)]
        push!(global_pair_vector, desired_sub_vector)
        offset = maximum(desired_sub_vector[lastindex(desired_sub_vector)])
    end
    return global_pair_vector
end


"""create_projection_vector(local_pair_vector,total_elements)
        compute the projection vector for redundant rotations
"""
function create_projection_vector(local_pair_vector, total_elements)
    global_pair_vector=get_global_pair(local_pair_vector)
    matrix=ones(total_elements,total_elements)
    kout = zeros(total_elements*(total_elements-1)÷2)
    ind = 1
    for sub_vector in global_pair_vector
        for (i, (x, y)) in enumerate(sub_vector) 
            matrix[x,y]=0
        end
    end
            
    for i in 1:total_elements
        for j in i+1:total_elements
            kout[ind] = matrix[i,j]
            ind += 1
        end
    end           
    return kout       
end

"""create_projection_matrix(projection vector)
        compute the projection matrix for redundant rotations
"""
function create_projection_matrix(vector)
    vector = reshape(vector, (length(vector), 1))  # Reshape the vector as a column vector
    projection_matrix = vector * transpose(vector)  # Compute the outer product

    return projection_matrix
end
"""get_rdm(ints, d1, k,clusters,fspace)
        computes the total rdm of the clusters
"""
function get_rdm(ints, d1, k,clusters,fspace)
    norb = n_orb(ints)
    K = unpack_gradient(k, norb)
    
    # Remove intracluster rotations
    for ci in clusters
        K[ci.orb_list, ci.orb_list] .= 0
    end
    #@show K
   
    Ui = exp(K)
    rdm1=orbital_rotation(d1,Ui)
    tmp_ints = orbital_rotation(ints,Ui)

    e, rdm1_dict, rdm2_dict = cmf_ci(tmp_ints, clusters, fspace, rdm1, 
                                     maxiter_d1 = 100, 
                                     maxiter_ci = 100, 
                                     tol_d1     = 1e-9, 
                                     tol_ci     = 1e-10, 
                                     verbose    = 0, 
                                     sequential = true)
    
    gd1, gd2 = assemble_full_rdm(clusters, rdm1_dict, rdm2_dict)
    
    return gd1,gd2
end

"""pack_hessian(Hessian,norb)
"""

function pack_hessian(H, norb)
    size(H) == (norb, norb, norb, norb) || throw(DimensionMismatch)
    hout = zeros(norb*(norb-1)÷2, norb*(norb-1)÷2)
    ind_row = 1
    for i in 1:norb
        for j in i+1:norb
            ind_col = 1
            for k in 1:norb
                for l in k+1:norb
                    hout[ind_row, ind_col] = H[i, j, k, l]
                    hout[ind_col,ind_row]=hout[ind_row,ind_col]
                    ind_col += 1
                end
            end
            ind_row += 1
        end
    end
    return hout
end

"""
    orbital_hessian_fd_fci_solve(ints, n_elec_a, n_elec_b, fspace, d; 
                                    verbose = 0,
                                    stepsize = 1e-4)
Compute orbital hessian with finite difference for FCI using ActiveSpaceSolvers
"""

function orbital_hessian_fd_fci_solve(ints,n_elec_a,n_elec_b, kappa, verbose = 0,stepsize = 1e-4)
    n = length(kappa)
    # error("here")
    hessian = zeros(n, n)

    function step_finite_difference!(ints, k)
        norb = n_orb(ints)
        K = unpack_gradient(k, norb)
       
        Ui = exp(K)
        
        tmp_ints = orbital_rotation(ints,Ui)
        
        ansatz = FCIAnsatz(norb, n_elec_a, n_elec_b)
        solver = SolverSettings(nroots=3, tol=1e-6, maxiter=100)
        solution = solve(tmp_ints, ansatz, solver)
        e = solution.energies
        return  e
    end
    for i in 1:n
    
        kappa_plus=deepcopy(kappa)
        kappa_plus[i]+=stepsize

        kappa_minus=deepcopy(kappa)
        kappa_minus[i]-=stepsize
        
        kappa_2plus=deepcopy(kappa)
        kappa_2plus[i] +=(2*stepsize)
        kappa_2minus=deepcopy(kappa)
        kappa_2minus[i]-=(2*stepsize)


        k0 = deepcopy(kappa)
        e0 = step_finite_difference!(ints,k0)
        k1 = deepcopy(kappa)
        k1[i] += stepsize
        e1 = step_finite_difference!(ints,k1)
        k2 = deepcopy(kappa)
        k2[i] -= stepsize
        e2 = step_finite_difference!(ints,k2)
        display(e2)
        grad = (e1[1]-e2[1])/(2*stepsize)
        hessian[i,i] = (e1[1] - 2*e0[1] + e2[1]) / (stepsize^2)

        @printf(" e0: %12.8f e1: %12.8f e2: %12.8f grad: %12.8f\n", e0[1], e1[1], e2[1], grad)
        
        for j in (i+1):n
            # Perturb parameters in both directions
            kappa_plus_plus = deepcopy(kappa)
            kappa_plus_plus[i] += stepsize
            kappa_plus_plus[j] += stepsize

            kappa_plus_minus = deepcopy(kappa)
            kappa_plus_minus[i] += stepsize
            kappa_plus_minus[j] -= stepsize

            kappa_minus_plus = deepcopy(kappa)
            kappa_minus_plus[i] -= stepsize
            kappa_minus_plus[j] += stepsize

            kappa_minus_minus = deepcopy(kappa)
            kappa_minus_minus[i] -= stepsize
            kappa_minus_minus[j] -= stepsize
            e_plus_plus=step_finite_difference!(ints,kappa_plus_plus)
            e_plus_minus=step_finite_difference!(ints,kappa_plus_minus)
            e_minus_plus=step_finite_difference!(ints,kappa_minus_plus)
            e_minus_minus=step_finite_difference!(ints,kappa_minus_minus)

            # Calculate finite difference approximation   
            hessian[i, j] = (e_plus_plus[1]-e_minus_plus[1] -e_plus_minus[1]+e_minus_minus[1])/(4*(stepsize^2))
            # Fill in the symmetric entry
            hessian[j, i] = hessian[i, j]
        end
    end

    return hessian
end
"""
    orbital_hessian_fd_fci_rdm(ints, n_elec_a, n_elec_b, fspace, d; 
                                    verbose = 0,
                                    stepsize = 1e-4)
Compute orbital hessian with finite difference for FCI using RDM
"""

function orbital_hessian_fd_fci_rdm(ints,n_elec_a,n_elec_b, kappa, verbose = 0,stepsize = 1e-4)
    n = length(kappa)
    # error("here")
    hessian = zeros(n, n)

    function step_fd!(ints, k)
        norb = n_orb(ints)
        K = unpack_gradient(k, norb)
       
        Ui = exp(K)
        
        tmp_ints = orbital_rotation(ints,Ui)
        
        ansatz = FCIAnsatz(norb, n_elec_a, n_elec_b)
        solver = SolverSettings(nroots=3, tol=1e-6, maxiter=100)
        solution = solve(tmp_ints, ansatz, solver)
        rdm1a, rdm1b,rdm2aa,rdm2bb,rdm2ab = ActiveSpaceSolvers.compute_1rdm_2rdm(solution, root=1)
        d1 = RDM1(rdm1a , rdm1b) 
        d2=RDM2(rdm2aa,rdm2ab,rdm2bb) 
        e = compute_energy(ints, d1, d2)
        return  e
    end
    for i in 1:n
        kappa_plus=deepcopy(kappa)
        kappa_plus[i]+=stepsize

        kappa_minus=deepcopy(kappa)
        kappa_minus[i]-=stepsize
        
        kappa_2plus=deepcopy(kappa)
        kappa_2plus[i] +=(2*stepsize)
        kappa_2minus=deepcopy(kappa)
        kappa_2minus[i]-=(2*stepsize)


        k0 = deepcopy(kappa)
        e0 = step_fd!(ints,k0)
        k1 = deepcopy(kappa)
        k1[i] += stepsize
        e1 = step_fd!(ints,k1)
        k2 = deepcopy(kappa)
        k2[i] -= stepsize
        e2 = step_fd!(ints,k2)
        grad = (e1[1]-e2[1])/(2*stepsize)
        hessian[i,i] = (e1[1] - 2*e0[1] + e2[1]) / (stepsize^2)

        @printf(" e0: %12.8f e1: %12.8f e2: %12.8f grad: %12.8f\n", e0, e1, e2, grad)
        
        for j in (i+1):n
            # Perturb parameters in both directions
            kappa_plus_plus = deepcopy(kappa)
            kappa_plus_plus[i] += stepsize
            kappa_plus_plus[j] += stepsize

            kappa_plus_minus = deepcopy(kappa)
            kappa_plus_minus[i] += stepsize
            kappa_plus_minus[j] -= stepsize

            kappa_minus_plus = deepcopy(kappa)
            kappa_minus_plus[i] -= stepsize
            kappa_minus_plus[j] += stepsize

            kappa_minus_minus = deepcopy(kappa)
            kappa_minus_minus[i] -= stepsize
            kappa_minus_minus[j] -= stepsize

            e_plus_plus=step_fd!(ints,kappa_plus_plus)
            e_plus_minus=step_fd!(ints,kappa_plus_minus)
            e_minus_plus=step_fd!(ints,kappa_minus_plus)
            e_minus_minus=step_fd!(ints,kappa_minus_minus)

            # Calculate finite difference approximation   
            hessian[i, j] = (e_plus_plus[1]-e_minus_plus[1] -e_plus_minus[1]+e_minus_minus[1])/(4*(stepsize^2))
            # Fill in the symmetric entry
            hessian[j, i] = hessian[i, j]
        end
    end

    return hessian
end

"""
    orbital_hessian_fd_cmf_rdm(ints, n_elec_a, n_elec_b, fspace, d; 
                                    verbose = 0,
                                    stepsize = 1e-4)
Compute orbital hessian with finite difference for cMF using RDM
"""
function orbital_hessian_fd_cmf_rdm(ints::InCoreInts{T},clusters,fspace, d1::RDM1{T},kappa,zero_intra_rots=true,verbose = 0,stepsize = 1e-5)where T
    n = length(kappa)
    # error("here")
    hessian = zeros(n, n)

    function step_fd_cmf!(ints, k,d1)
        norb = n_orb(ints)
        K = unpack_gradient(k, norb)
        if zero_intra_rots
            # Remove intracluster rotations
            for ci in clusters
                K[ci.orb_list, ci.orb_list] .= 0
            end
        end
       
        Ui = exp(K)
        
        tmp_ints = orbital_rotation(ints,Ui)

        e, rdm1_dict, rdm2_dict = cmf_ci(tmp_ints, clusters, fspace, d1, 
                                         maxiter_d1 = 100, 
                                         maxiter_ci = 100, 
                                         tol_d1     = 1e-9, 
                                         tol_ci     = 1e-10, 
                                         verbose    = 0, 
                                         sequential = true)
        
        gd1, gd2 = assemble_full_rdm(clusters, rdm1_dict, rdm2_dict) 
        e = compute_energy(tmp_ints, gd1, gd2)
        return  e
    end
    for i in 1:n
        kappa_plus=deepcopy(kappa)
        kappa_plus[i]+=stepsize

        kappa_minus=deepcopy(kappa)
        kappa_minus[i]-=stepsize
        
        kappa_2plus=deepcopy(kappa)
        kappa_2plus[i] +=(2*stepsize)
        kappa_2minus=deepcopy(kappa)
        kappa_2minus[i]-=(2*stepsize)


        k0 = deepcopy(kappa)
        e0 = step_fd_cmf!(ints,k0,d1)
        k1 = deepcopy(kappa)
        k1[i] += stepsize
        e1 = step_fd_cmf!(ints,k1,d1)
        k2 = deepcopy(kappa)
        k2[i] -= stepsize
        e2 = step_fd_cmf!(ints,k2,d1)
        grad = (e1-e2)/(2*stepsize)
        hessian[i,i] = (e1 - 2*e0 + e2) / (stepsize^2)

        @printf(" e0: %12.8f e1: %12.8f e2: %12.8f grad: %12.8f\n", e0, e1, e2, grad)
        
        for j in (i+1):n
            # Perturb parameters in both directions
            kappa_plus_plus = deepcopy(kappa)
            kappa_plus_plus[i] += stepsize
            kappa_plus_plus[j] += stepsize

            kappa_plus_minus = deepcopy(kappa)
            kappa_plus_minus[i] += stepsize
            kappa_plus_minus[j] -= stepsize

            kappa_minus_plus = deepcopy(kappa)
            kappa_minus_plus[i] -= stepsize
            kappa_minus_plus[j] += stepsize

            kappa_minus_minus = deepcopy(kappa)
            kappa_minus_minus[i] -= stepsize
            kappa_minus_minus[j] -= stepsize

            e_plus_plus=step_fd_cmf!(ints,kappa_plus_plus,d1)
            e_plus_minus=step_fd_cmf!(ints,kappa_plus_minus,d1)
            e_minus_plus=step_fd_cmf!(ints,kappa_minus_plus,d1)
            e_minus_minus=step_fd_cmf!(ints,kappa_minus_minus,d1)

            # Calculate finite difference approximation   
            hessian[i, j] = (e_plus_plus-e_minus_plus-e_plus_minus+e_minus_minus)/(4*(stepsize^2))
            # Fill in the symmetric entry
            hessian[j, i] = hessian[i, j]
        end
    end

    return hessian
end





"""
    orbital_hessian_finite_difference(ints, clusters, kappa, fspace, d; 
                                    ci_conv = 1e-8, 
                                    verbose = 1,
                                    stepsize = 1e-6)
Compute orbital hessian with finite difference for cMF using orbital energy
"""
function orbital_hessian_finite_difference(ints, clusters, kappa, fspace, ansatze,d::RDM1; ci_conv = 1e-8, verbose = 0,stepsize = 1e-5)
    n = length(kappa)
    # error("here")
    hessian = zeros(n, n)
    for i in 1:n
        kappa_plus=deepcopy(kappa)
        kappa_plus[i]+=stepsize

        kappa_minus=deepcopy(kappa)
        kappa_minus[i]-=stepsize
        #hessian[i,i]=(orbital_objective_function(ints, clusters, kappa_plus, fspace, d, ci_conv = ci_conv, verbose = verbose)-2*orbital_objective_function(ints, clusters, kappa, fspace, d, ci_conv = ci_conv, verbose = verbose)+orbital_objective_function(ints, clusters, kappa_minus, fspace, d, ci_conv = ci_conv, verbose = verbose))/(stepsize^2)
        kappa_2plus=deepcopy(kappa)
        kappa_2plus[i] +=(2*stepsize)
        kappa_2minus=deepcopy(kappa)
        kappa_2minus[i]-=(2*stepsize)


        k0 = deepcopy(kappa)
        e0 = orbital_objective_function(ints, clusters, k0, fspace,ansatze, d, ci_conv=ci_conv, verbose=verbose) 

        k1 = deepcopy(kappa)
        k1[i] += stepsize
        e1 = orbital_objective_function(ints, clusters, k1, fspace, ansatze,d, ci_conv=ci_conv, verbose=verbose) 
        
        k2 = deepcopy(kappa)
        k2[i] -= stepsize
        e2 = orbital_objective_function(ints, clusters, k2, fspace, ansatze,d, ci_conv=ci_conv, verbose=verbose) 
        
        grad = (e1-e2)/(2*stepsize)
        hessian[i,i] = (e1 - 2*e0 + e2) / (stepsize^2)

        # @printf(" e0: %12.8f e1: %12.8f e2: %12.8f grad: %12.8f\n", e0, e1, e2, grad)
        
        for j in (i+1):n
            # Perturb parameters in both directions
            kappa_plus_plus = deepcopy(kappa)
            kappa_plus_plus[i] += stepsize
            kappa_plus_plus[j] += stepsize

            kappa_plus_minus = deepcopy(kappa)
            kappa_plus_minus[i] += stepsize
            kappa_plus_minus[j] -= stepsize

            kappa_minus_plus = deepcopy(kappa)
            kappa_minus_plus[i] -= stepsize
            kappa_minus_plus[j] += stepsize

            kappa_minus_minus = deepcopy(kappa)
            kappa_minus_minus[i] -= stepsize
            kappa_minus_minus[j] -= stepsize

            # Calculate finite difference approximation   
            hessian[i, j] = (orbital_objective_function(ints, clusters, kappa_plus_plus, fspace, ansatze,d, ci_conv = ci_conv, verbose = verbose) -
                            orbital_objective_function(ints, clusters, kappa_plus_minus, fspace, ansatze,d, ci_conv = ci_conv, verbose = verbose) -
                            orbital_objective_function(ints, clusters, kappa_minus_plus, fspace, ansatze,d, ci_conv = ci_conv, verbose = verbose) +
                            orbital_objective_function(ints, clusters, kappa_minus_minus, fspace, ansatze,d, ci_conv = ci_conv, verbose = verbose))/(4*(stepsize^2))
            # Fill in the symmetric entry
            hessian[j, i] = hessian[i, j]
        end
    end

    return hessian
end


"""
    orbital_hessian_numerical(ints, clusters, kappa, fspace, d; 
                                    ci_conv = 1e-8, 
                                    verbose = 1,
                                    stepsize = 1e-6)
Compute orbital hessian with finite difference for cMF using orbital gradient 
"""
function orbital_hessian_numerical(ints, clusters, kappa, fspace, ansatze,d::RDM1; verbose = 0,step_size = 5e-5,zero_intra_rots = true,maxiter_ci = 100, maxiter_d1 = 100, tol_d1 = 1e-6, tol_ci = 1e-8,sequential = false)
    n = length(kappa)
    #error("here.....")
    hessian = zeros(n, n)
    function step_numerical!(ints, d1, k)
        norb = n_orb(ints)
        K = unpack_gradient(k, norb)
        # if zero_intra_rots
        #     # Remove intracluster rotations
        #     for ci in clusters
        #         K[ci.orb_list, ci.orb_list] .= 0
        #     end
        # end
       
        Ui = exp(K)
        
        tmp_ints = orbital_rotation(ints,Ui)

        e, rdm1_dict, rdm2_dict = cmf_ci(tmp_ints, clusters, fspace,ansatze, d1, 
                                         maxiter_d1 = maxiter_d1, 
                                         maxiter_ci = maxiter_ci, 
                                         tol_d1     = tol_d1, 
                                         tol_ci     = tol_ci, 
                                         verbose    = 0, 
                                         sequential = sequential)
        
        gd1, gd2 = assemble_full_rdm(clusters, rdm1_dict, rdm2_dict)
        G = build_orbital_gradient(tmp_ints, gd1, gd2)
        # G=orbital_gradient_numerical(ints,clusters,k,fspace,gd1)
        return  G
    end
    
    norb = n_orb(ints)
    #central difference 
    for i in 1:n
        println(i)
        x_plus_i = deepcopy(kappa)
        x_minus_i = deepcopy(kappa)
        x_plus_i[i] += step_size
        x_minus_i[i] -= step_size
        g_plus_i=step_numerical!(ints,d,x_plus_i)
        g_minus_i=step_numerical!(ints,d,x_minus_i)
        
        gnum = (g_plus_i .- g_minus_i)./(2*step_size)
        # display(gnum)
        for j in 1:n
            hessian[i,j] = gnum[j]
            #error("dfdg")
        end

        
    end
    # display(hessian)
    return hessian
end