using QCBase  
using RDM  
using ClusterMeanField  
using LinearAlgebra
using Printf
using Test
using NPZ
using InCoreIntegrals
using ActiveSpaceSolvers


@testset "CMF" begin
    #h0 = npzread("h6_sto3g/h0.npy")
    #h1 = npzread("h6_sto3g/h1.npy")
    #h2 = npzread("h6_sto3g/h2.npy")
    
    atoms = []
    push!(atoms,Atom(1,"H",[0,0,0]))
    push!(atoms,Atom(2,"H",[1,0,0]))
    push!(atoms,Atom(3,"H",[2,0,0]))
    push!(atoms,Atom(4,"H",[3,0,0]))
    push!(atoms,Atom(5,"H",[4,0,0]))
    push!(atoms,Atom(6,"H",[5,0,0]))
    #basis = "6-31g"
    basis = "sto-3g"

    mol     = Molecule(0,1,atoms,basis)
    mf = ClusterMeanField.pyscf_do_scf(mol)
    nbas = size(mf.mo_coeff)[1]
    ints = ClusterMeanField.pyscf_build_ints(mol,mf.mo_coeff, zeros(nbas,nbas));
    e_fci, d1a_fci, d1b_fci,d2_fci = ClusterMeanField.pyscf_fci(ints,3,3)
    # @printf(" FCI Energy: %12.8f\n", e_fci)

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

    clusters    = [(1:2),(3:4),(5:6)]
    init_fspace = [(1,1),(1,1),(1,1)]

    clusters = [MOCluster(i,collect(clusters[i])) for i = 1:length(clusters)]
    display(clusters)

    rdm1 = RDM1(rdm_mf, rdm_mf)

    e_fci = -3.155304800477
    e_scf = -3.09169726403968
   
    sol = solve(ints, FCIAnsatz(6,3,3), SolverSettings())
    display(sol)
    
    clusters    = [(1:2),(3:4),(5:6)]
    init_fspace = [(1,1),(1,1),(1,1)]

    clusters = [MOCluster(i,collect(clusters[i])) for i = 1:length(clusters)]
    display(clusters)

    f1 = cmf_ci(ints, clusters, init_fspace, rdm1, 
                        verbose=1, sequential=false)
    
    @test isapprox(f1[1], -2.97293813654926351, atol=1e-10)
    
    e_cmf, U = cmf_oo(ints, clusters, init_fspace, rdm1, 
                              verbose=0, gconv=1e-6, method="cg",sequential=true)
    @test isapprox(e_cmf, -3.205983033016, atol=1e-10)
    
    Ccmf = Cl*U
    ClusterMeanField.pyscf_write_molden(mol,Ccmf,filename="cmf.molden")
  
    

end
    
#@testset "CMF open shell" begin
# 
    atoms = []
    push!(atoms,Atom(1,"H",[0,0,0]))
    push!(atoms,Atom(2,"H",[1,0,0]))
    push!(atoms,Atom(3,"H",[2,0,0]))
    push!(atoms,Atom(4,"H",[3,0,0]))
    push!(atoms,Atom(5,"H",[4,0,0]))
    push!(atoms,Atom(6,"H",[5,0,0]))
    #basis = "6-31g"
    basis = "sto-3g"

    mol     = Molecule(0,1,atoms,basis)
    mf = ClusterMeanField.pyscf_do_scf(mol)
    nbas = size(mf.mo_coeff)[1]
    ints = ClusterMeanField.pyscf_build_ints(mol,mf.mo_coeff, zeros(nbas,nbas));
    e_fci, d1a_fci, d1b_fci,d2_fci = ClusterMeanField.pyscf_fci(ints,3,3)
    # @printf(" FCI Energy: %12.8f\n", e_fci)

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

    rdm1 = RDM1(rdm_mf, rdm_mf)

    e_fci = -3.155304800477
    e_scf = -3.09169726403968
   
    sol = solve(ints, FCIAnsatz(6,3,3), SolverSettings())
    display(sol)
   
    clusters    = [(1:3),(4:6)]
    init_fspace = [(2,1),(1,2)]
    clusters    = [(1:2),(3:4),(5:6)]
    init_fspace = [(1,1),(1,1),(1,1)]

    clusters = [MOCluster(i,collect(clusters[i])) for i = 1:length(clusters)]
    display(clusters)

    n = n_orb(ints)
    k = zeros(n*(n-1)÷2)
    
    f1 = cmf_ci(ints, clusters, init_fspace, rdm1, 
                        verbose=1, sequential=false)
   
    
    g_num = ClusterMeanField.orbital_gradient_numerical(ints, clusters, k, init_fspace, rdm1) 
    g_anl = ClusterMeanField.orbital_gradient_analytical(ints, clusters, k, init_fspace, rdm1) 
    println(" Here is the error:")
    display(norm(g_num-g_anl))
   
    d1dict = f1[2]
    d2dict = f1[3]

    d1, d2 = ClusterMeanField.assemble_full_rdm(clusters, d1dict, d2dict)

    rdm1 = ssRDM1(d1)
    rdm2 = ssRDM2(d2)


    display(rdm1)
    display(tr(rdm1.rdm))
    display(compute_energy(ints, rdm1, rdm2))
    display(compute_energy(ints, d1, d2))
    display(compute_energy(ints, d1dict, d2dict, clusters))
    println(" Here is the error:")
    display(norm(g_num-g_anl))
    #@test isapprox(f1[1], -2.97293813654926351, atol=1e-10)
    
    e_cmf, U = cmf_oo(ints, clusters, init_fspace, d1, 
                              verbose=0, gconv=1e-6, method="cg",sequential=true)
    #@test isapprox(e_cmf, -3.205983033016, atol=1e-10)
    
    Ccmf = Cl*U
    ClusterMeanField.pyscf_write_molden(mol,Ccmf,filename="cmf.molden")
  
    

#end
