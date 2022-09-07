
using ClusterMeanField  
using LinearAlgebra
using Printf
using Test
using NPZ
using InCoreIntegrals
using ActiveSpaceSolvers
using Random

function run_o()
# 
    atoms = []
    push!(atoms,Atom(1,"O",[0,0,0]))
    #basis = "6-31g"
    basis = "sto-3g"

    mol     = Molecule(0,1,atoms,basis)
    mf = ClusterMeanField.pyscf_do_scf(mol)
    nbas = size(mf.mo_coeff)[1]
    ints = ClusterMeanField.pyscf_build_ints(mol,mf.mo_coeff, zeros(nbas,nbas));
    e_fci, d1a_fci, d1b_fci,d2_fci = ClusterMeanField.pyscf_fci(ints,3,3)
     @printf(" FCI Energy: %12.8f\n", e_fci)

    ClusterMeanField.pyscf_write_molden(mol,mf.mo_coeff,filename="scf.molden")
    C = mf.mo_coeff
    rdm_mf = C[:,1:2] * C[:,1:2]'
    Cl = ClusterMeanField.localize(mf.mo_coeff,"lowdin",mf)
    ClusterMeanField.pyscf_write_molden(mol,Cl,filename="lowdin.molden")
    S = ClusterMeanField.get_ovlp(mf)
    U =  C' * S * Cl
    println(" Build Integrals")
    flush(stdout)
    ints = orbital_rotation(ints,U)
    println(" done.")
    flush(stdout)

    clusters    = [(1,),(2,5),(3,4)]
    init_fspace = [(1,1),(1,1),(1,1)]

    clusters = [Cluster(i,collect(clusters[i])) for i = 1:length(clusters)]
    display(clusters)

    rdm1 = zeros(size(ints.h1))
    rdm1a = rdm_mf
    rdm1b = rdm_mf
    
    sol = solve(ints, FCIAnsatz(5,3,3), SolverSettings())
    display(sol)

    f1 = cmf_ci(ints, clusters, init_fspace, rdm1a, rdm1b, 
                        verbose=1, sequential=false)
    
    
    print("\n Now do high-spin case\n")
    init_fspace = [(1,1),(2,0),(1,1)]
    f2 = cmf_ci(ints, clusters, init_fspace, rdm1a, rdm1b, 
                        verbose=1, sequential=false)
    
    @test isapprox(f2[1], f1[1], atol=1e-6)
    #e_cmf, U = cmf_oo(ints, clusters, init_fspace, rdm1, rdm1, 
    #                          verbose=0, gconv=1e-7, method="cg",sequential=true)
    
    #Ccmf = Cl*U
    #ClusterMeanField.pyscf_write_molden(mol,Ccmf,filename="cmf.molden")
  
end

function run_h6()
# 
    atoms = []
    atoms = []
    push!(atoms,Atom(1,"H",[0,0,0]))
    push!(atoms,Atom(2,"H",[1,0,0]))
    push!(atoms,Atom(3,"H",[2,0,0]))
    push!(atoms,Atom(4,"H",[3,0,0]))
    push!(atoms,Atom(5,"H",[4,0,0]))
    push!(atoms,Atom(6,"H",[5,0,0]))
    basis = "sto-3g"

    mol     = Molecule(0,1,atoms,basis)
    mf = ClusterMeanField.pyscf_do_scf(mol)
    nbas = size(mf.mo_coeff)[1]
    ints = ClusterMeanField.pyscf_build_ints(mol,mf.mo_coeff, zeros(nbas,nbas));
    e_fci, d1a_fci, d1b_fci,d2_fci = ClusterMeanField.pyscf_fci(ints,3,3)
     @printf(" FCI Energy: %12.8f\n", e_fci)

    ClusterMeanField.pyscf_write_molden(mol,mf.mo_coeff,filename="scf.molden")
    C = mf.mo_coeff
    rdm_mf = C[:,1:2] * C[:,1:2]'
    Cl = ClusterMeanField.localize(mf.mo_coeff,"lowdin",mf)
    ClusterMeanField.pyscf_write_molden(mol,Cl,filename="lowdin.molden")
    S = ClusterMeanField.get_ovlp(mf)
    U =  C' * S * Cl
    println(" Build Integrals")
    flush(stdout)
    ints = orbital_rotation(ints,U)
    println(" done.")
    flush(stdout)

    clusters    = [(1:3),(4:6)]
    init_fspace = [(2,2),(2,2)]
        
    clusters    = [(1,2),(3,4),(5:6)]
    init_fspace = [(1,1),(1,1),(1,1)]

    clusters = [Cluster(i,collect(clusters[i])) for i = 1:length(clusters)]
    display(clusters)

    rdm1 = zeros(size(ints.h1))
    rdm1a = rdm_mf
    rdm1b = rdm_mf
    
    sol = solve(ints, FCIAnsatz(6,4,2), SolverSettings())
    display(sol)

    f1 = cmf_ci(ints, clusters, init_fspace, rdm1a, rdm1b, 
                        verbose=1, sequential=false)
  
    Random.seed!(2)
    norb = n_orb(ints)
    kappa = rand(norb,norb).*.0
    #kappa[1,2] = 2*pi 
    kappa = kappa - kappa'
    U = exp(kappa)
    kappa = ClusterMeanField.pack_gradient(kappa, norb)
    
    
    etmp = ClusterMeanField.orbital_objective_function(ints, clusters, kappa, init_fspace, rdm1, rdm1, ci_conv=1e-10)
    print("nick E ")
    display(etmp)
    g_anl = ClusterMeanField.orbital_gradient_analytical(ints, clusters, kappa, init_fspace, rdm1, rdm1, ci_conv=1e-10)
    display(ClusterMeanField.unpack_gradient(g_anl, norb))
    g_num = ClusterMeanField.orbital_gradient_numerical(ints, clusters, kappa, init_fspace, rdm1, rdm1, stepsize=1e-5, ci_conv = 1e-10)
    display(ClusterMeanField.unpack_gradient(g_num, norb))

    println()
    g_anl = ClusterMeanField.unpack_gradient(g_anl, norb)
    g_num = ClusterMeanField.unpack_gradient(g_num, norb)
  
    display(U)
    display(norm(g_anl - g_num))
    display(sort(imag(eigvals(g_num))))
    display(sort(imag(eigvals(g_anl))))
    #e_cmf, U = cmf_oo(ints, clusters, init_fspace, rdm1, rdm1, 
    #                          verbose=0, gconv=1e-7, method="cg",sequential=false, max_iter_oo=200)
   
    return
    if true 
        # Now do closed shell clusters
        clusters    = [(1,2),(3,4),(5:6)]
        init_fspace = [(1,1),(1,1),(0,0)]
        clusters = [Cluster(i,collect(clusters[i])) for i = 1:length(clusters)]
        display(clusters)
    end
    
    g_anl = ClusterMeanField.orbital_gradient_analytical(ints, clusters, kappa, init_fspace, rdm1, rdm1, verbose=0)
    g_num = ClusterMeanField.orbital_gradient_numerical(ints, clusters, kappa, init_fspace, rdm1, rdm1, stepsize=1e-7, gconv = 1e-9, verbose=0)
   
    display(norm(g_anl - g_num))
    
    e_cmf, U = cmf_oo(ints, clusters, init_fspace, rdm1, rdm1, 
                              verbose=0, gconv=1e-7, method="gd",sequential=false, max_iter_oo=200)
    Ccmf = Cl*U
    ints = orbital_rotation(ints,U)
    
    e_cmf, U = cmf_oo(ints, clusters, init_fspace, rdm1, rdm1, 
                              verbose=0, gconv=1e-7, method="bfgs",sequential=false, max_iter_oo=200)

    ints = orbital_rotation(ints,U)
    Ccmf = Ccmf*U
    ClusterMeanField.pyscf_write_molden(mol,Ccmf,filename="cmf.molden")
  
end

run_h6()
