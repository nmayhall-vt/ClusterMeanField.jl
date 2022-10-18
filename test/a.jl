using QCBase  
using RDM  
using ClusterMeanField  
using LinearAlgebra
using Printf
using Test
using NPZ
using InCoreIntegrals
using ActiveSpaceSolvers

    
    atoms = []
    push!(atoms,Atom(1,"H",[0,0,0]))
    push!(atoms,Atom(2,"H",[1,0,0]))
    push!(atoms,Atom(3,"H",[2,0,0]))
    push!(atoms,Atom(4,"H",[3,1,0]))
    push!(atoms,Atom(5,"H",[4,1,0]))
    push!(atoms,Atom(6,"H",[5,1,0]))
    #basis = "6-31g"
    basis = "sto-3g"

    na = 4
    nb = 3

    mol     = Molecule(-1,abs(na-nb)+1,atoms,basis)
    mf = ClusterMeanField.pyscf_do_scf(mol)
    nbas = size(mf.mo_coeff)[1]
    ints = ClusterMeanField.pyscf_build_ints(mol,mf.mo_coeff, zeros(nbas,nbas));


    e_fci, d1a_fci, d1b_fci,d2_fci = ClusterMeanField.pyscf_fci(ints, na, nb)
    @printf(" HF  Energy: %12.8f\n", mf.e_tot)
    @printf(" FCI Energy: %12.8f\n", e_fci+ints.h0)

    ClusterMeanField.pyscf_write_molden(mol,mf.mo_coeff,filename="scf.molden")

    C = mf.mo_coeff
    S = mf.get_ovlp()
    
    display(mf.make_rdm1())

    d1_fci = RDM1(d1a_fci,d1b_fci)
    
    d1_fci = ssRDM1(d1a_fci+d1b_fci)
    d2_fci = ssRDM2(d2_fci)

    display(tr(mf.make_rdm1()[2] * S ))
    error("here")
    rdm_mf_a = C' * S * mf.make_rdm1()[1] * S * C
    rdm_mf_b = C' * S * mf.make_rdm1()[2] * S * C
    rdm1 = RDM1(rdm_mf_a, rdm_mf_b)
    display(tr(rdm1.a*S))
   
    @printf(" Should be E(HF):  %12.8f\n", compute_energy(ints, rdm1, RDM2(rdm1)))
    @printf(" Should be E(HF):  %12.8f\n", compute_energy(ints, rdm1))
    @printf(" Should be E(FCI): %12.8f\n", compute_energy(ints, d1_fci, d2_fci))
    
    Cl = ClusterMeanField.localize(mf.mo_coeff,"lowdin",mf)
    ClusterMeanField.pyscf_write_molden(mol,Cl,filename="lowdin.molden")
    S = ClusterMeanField.get_ovlp(mf)
    U =  C' * S * Cl
    println(" Build Integrals")
    flush(stdout)
    ints = orbital_rotation(ints,U)
    rdm1 = orbital_rotation(rdm1,U)
    d1_fci = orbital_rotation(d1_fci,U)
    d2_fci = orbital_rotation(d2_fci,U)
    println(" done.")
    flush(stdout)


    @printf(" Should be E(HF):  %12.8f\n", compute_energy(ints, rdm1, RDM2(rdm1)))
    @printf(" Should be E(FCI): %12.8f\n", compute_energy(ints, d1_fci, d2_fci))
    
    e_fci = -3.155304800477
    e_scf = -3.09169726403968
   
    sol = solve(ints, FCIAnsatz(6,3,3), SolverSettings(nroots=1, tol=1e-8))
    display(sol)
    d1a, d1b, d2aa, d2bb, d2ab = compute_1rdm_2rdm(sol) 
    
    d1_fci = RDM1(d1a, d1b)
    d2_fci = RDM2(d2aa, d2ab, d2bb)

    clusters    = [(1:3),(4:6)]
    init_fspace = [(2,1),(1,2)]

    clusters = [MOCluster(i,collect(clusters[i])) for i = 1:length(clusters)]
    display(clusters)

    n = n_orb(ints)
    
    ecmf, d1dict, d2dict = cmf_ci(ints, clusters, init_fspace, rdm1, 
                        verbose=1, sequential=false)
    
    # 
    #   Test 1 and 2 RDM consistency 
    if false
        d1, d2 = ClusterMeanField.assemble_full_rdm(clusters, d1dict, d2dict)

        display(d1)
        display(RDM1(d2))
        @test isapprox(norm(d1.a-RDM1(d2).a), 0, atol=1e-8)
        @test isapprox(norm(d1.b-RDM1(d2).b), 0, atol=1e-8)
    end

  
    #
    #   Test numerical gradients
    d1, d2 = ClusterMeanField.assemble_full_rdm(clusters, d1dict, d2dict)
    rdm1 = ssRDM1(d1)
    rdm2 = ssRDM2(d2)
    

    e_cmf, U, d1 = ClusterMeanField.cmf_oo_diis(ints, clusters, init_fspace, d1,
                           maxiter_oo   = 200,
                           maxiter_ci   = 200,
                           maxiter_d1   = 200,
                           max_ss_size  = 8,
                           verbose      = 0,
                           tol_oo       = 1e-7,
                           tol_d1       = 1e-9,
                           tol_ci       = 1e-11,
                           sequential   = false,
                           alpha        = .2,
                           use_pyscf    = false,
                           diis_start   = 1)


