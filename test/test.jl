
using ClusterMeanField  
using LinearAlgebra
using Printf
using Test
using NPZ
using InCoreIntegrals
using ActiveSpaceSolvers

function run()
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

run()
