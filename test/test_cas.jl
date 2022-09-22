using ClusterMeanField  
using LinearAlgebra
using Printf
using Test
using NPZ
using InCoreIntegrals
using ActiveSpaceSolvers
using PyCall

#@testset "cas" begin


    atoms = []
    push!(atoms,Atom(1,"H",[0,0,0]))
    push!(atoms,Atom(2,"H",[1,0,0]))
    push!(atoms,Atom(3,"H",[2,0,0]))
    push!(atoms,Atom(4,"H",[3,0,0]))
    push!(atoms,Atom(5,"H",[4,0,0]))
    push!(atoms,Atom(6,"H",[5,0,0]))
    push!(atoms,Atom(7,"H",[6,0,0]))
    push!(atoms,Atom(8,"H",[7,0,0]))
    #basis = "6-31g"
    basis = "sto-3g"

    mol     = Molecule(1,2,atoms,basis)
    pymol = ClusterMeanField.make_pyscf_mole(mol)
    #mf = ClusterMeanField.pyscf_do_scf(mol)
    pyscf = pyimport("pyscf")
    mf = pyscf.scf.ROHF(pymol).run()

    mcscf = pyimport("pyscf.mcscf")
    cas = mcscf.CASCI(mf, 4, 3 ).kernel()

    nbas = size(mf.mo_coeff)[1]
    ints = ClusterMeanField.pyscf_build_ints(mol,mf.mo_coeff, zeros(nbas,nbas));
    ClusterMeanField.pyscf_fci(ints, 4,3)


    clusters    = [(1,2),(3:6),(7,8)]
    init_fspace = [(2,2),(2,1),(0,0)]
    clusters = [MOCluster(i,collect(clusters[i])) for i = 1:length(clusters)]
    display(clusters)
    
    sol = solve(ints, FCIAnsatz(8,4,3), SolverSettings())
    display(sol)

    display(mf.mo_coeff)
    display(size(mf.make_rdm1()))
    rdm_mf = mf.make_rdm1()
    rdm_mfa = rdm_mf[1,:,:]
    rdm_mfb = rdm_mf[2,:,:]
    rdm_mfa = mf.mo_coeff'*mf.get_ovlp()*rdm_mfa*mf.get_ovlp()*mf.mo_coeff
    rdm_mfb = mf.mo_coeff'*mf.get_ovlp()*rdm_mfb*mf.get_ovlp()*mf.mo_coeff
    rdm1 = RDM1(rdm_mfa, rdm_mfb)
    display(rdm1)

    
    f1 = cmf_ci(ints, clusters, init_fspace, rdm1, 
                        verbose=1, sequential=false)
    @printf(" CMF Energy: %12.8f\n", f1[1])


    d1, d2 = ClusterMeanField.assemble_full_rdm(clusters, f1[2], f1[3])
    g_anl2 = build_orbital_gradient(ints, d1, d2)
    display(round.(ClusterMeanField.unpack_gradient(g_anl2, nbas), digits=7))
#end

