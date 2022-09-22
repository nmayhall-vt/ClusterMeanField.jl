using QCBase  
using RDM  
using ClusterMeanField  
using LinearAlgebra
using Printf
using Test
using NPZ
using InCoreIntegrals
using ActiveSpaceSolvers

function numgrad(ints, d1::RDM1, d2::RDM2)


    @printf(" Energy: %12.8f\n", compute_energy(ints, d1, d2))

    no = n_orb(ints)
    k = RDM.pack_gradient(zeros(no,no), no)
    grad = deepcopy(k)

    stepsize=1e-5
    for i in 1:length(k)
        ki = zeros(no*(no-1)÷2) 
        ki[i] += stepsize
        Ki = RDM.unpack_gradient(ki, no)
        U = exp(Ki)
        intsi = orbital_rotation(ints,U)
        e1 = compute_energy(intsi, d1, d2)
        
        ki = zeros(no*(no-1)÷2) 
        ki[i] -= stepsize
        Ki = RDM.unpack_gradient(ki, no)
        U = exp(Ki)
        intsi = orbital_rotation(ints,U)
        e2 = compute_energy(intsi, d1, d2)
        
        grad[i] = (e1-e2)/(2*stepsize)
    end
    println(" Numerical Gradient: ")
    display(norm(grad))
    #display(grad)
    
    println(" Analytical Gradient: ")
    g = build_orbital_gradient(ints, d1, d2)
    display(norm(g))
    @printf("   Error: %12.8f\n",norm(g-grad))
    #display(g)

    
    println()
    g = build_orbital_gradient(ints, ssRDM1(d1), ssRDM2(d2))
    #display(g)
    println(" Analytical Gradient: ")
    display(norm(g))
    @printf("   Error: %12.8f\n",norm(g-grad))
    println(" Now compare gradients:")
    println(" Numerical:")
    display(round.(ClusterMeanField.unpack_gradient(grad, no), digits=7))
    println(" Analytical:")
    display(round.(ClusterMeanField.unpack_gradient(g, no), digits=7))
    #error("here")
end

if false 
#@testset "CMF" begin
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
    @printf(" HF  Energy: %12.8f\n", mf.e_tot)
    @printf(" FCI Energy: %12.8f\n", e_fci+ints.h0)

    ClusterMeanField.pyscf_write_molden(mol,mf.mo_coeff,filename="scf.molden")

    C = mf.mo_coeff

    d1_fci = RDM1(d1a_fci,d1b_fci)
    
    d1_fci = ssRDM1(d1a_fci+d1b_fci)
    d2_fci = ssRDM2(d2_fci)

    rdm_mf = mf.mo_coeff'*mf.get_ovlp()*mf.make_rdm1()*mf.get_ovlp()*mf.mo_coeff / 2
    rdm1 = RDM1(rdm_mf, rdm_mf)
    
    @printf(" Should be E(HF):  %12.8f\n", compute_energy(ints, rdm1, RDM2(rdm1)))
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

    clusters    = [(1:2),(3:4),(5:6)]
    init_fspace = [(1,1),(1,1),(1,1)]
    clusters    = [(1:3),(4:6)]
    init_fspace = [(2,1),(1,2)]
    clusters    = [(1:3),(4:6)]
    init_fspace = [(2,1),(2,1)]

    clusters = [MOCluster(i,collect(clusters[i])) for i = 1:length(clusters)]
    display(clusters)

    n = n_orb(ints)
    k = zeros(n*(n-1)÷2)
    
    f1 = cmf_ci(ints, clusters, init_fspace, rdm1, 
                        verbose=1, sequential=false)
   
    
    g_num = ClusterMeanField.orbital_gradient_numerical(ints, clusters, k, init_fspace, rdm1, stepsize=1e-5) 
    g_anl = ClusterMeanField.orbital_gradient_analytical(ints, clusters, k, init_fspace, rdm1) 
    println(" Here is the error:")
    display(norm(g_num-g_anl))
   
    d1dict = f1[2]
    d2dict = f1[3]

    d1, d2 = ClusterMeanField.assemble_full_rdm(clusters, d1dict, d2dict)

    rdm1 = ssRDM1(d1)
    rdm2 = ssRDM2(d2)

    display(d1)
    display(RDM1(d2))
    @test isapprox(norm(d1.a-RDM1(d2).a), 0, atol=1e-8)
    @test isapprox(norm(d1.b-RDM1(d2).b), 0, atol=1e-8)

    println(" These should match")
    display(compute_energy(ints, rdm1, rdm2))
    display(compute_energy(ints, d1, d2))
    display(compute_energy(ints, d1dict, d2dict, clusters))
    
    @test isapprox(compute_energy(ints, rdm1, rdm2), compute_energy(ints, d1, d2), atol=1e-12)
    @test isapprox(compute_energy(ints, rdm1, rdm2), compute_energy(ints, d1dict, d2dict, clusters), atol=1e-12)

#    #g_anl2 = build_orbital_gradient(ints, d1, d2)
#    @printf(" Shold be FCI: %12.8f\n",compute_energy(ints, d1_fci, d2_fci))
#    g_anl2 = ClusterMeanField.mybuild_orbital_gradient(ints, d1_fci, d2_fci)
#    println(" Here is the error:")
#    display(norm(g_num-g_anl2))
#    display(round.(ClusterMeanField.unpack_gradient(g_num, n), digits=7))
#    display(round.(ClusterMeanField.unpack_gradient(g_anl2, n), digits=7))
#    #display(g_num)
#    #display(g_anl2)
   

    g_anl2 = build_orbital_gradient(ints, d1, d2)
    println(" Here is the error:")
    display(norm(g_num-g_anl2))
   
    display(round.(ClusterMeanField.unpack_gradient(g_num, n), digits=7))
    display(round.(ClusterMeanField.unpack_gradient(g_anl2, n), digits=7))

    #f1 = cmf_ci(ints, clusters, init_fspace, d1, verbose=1, sequential=false)
    #d1 = f1[2]
    #d2 = f1[3]
    #d1,d2 = ClusterMeanField.assemble_full_rdm(clusters, d1, d2)
    numgrad(ints, d1, d2)

    return
    #e_cmf, U = cmf_oo(ints, clusters, init_fspace, d1, 
    #                          verbose=0, gconv=1e-6, method="gd",sequential=true)
    #@test isapprox(e_cmf, -3.050480022999, atol=1e-10)
    
    #Ccmf = Cl*U
    #ClusterMeanField.pyscf_write_molden(mol,Ccmf,filename="cmf.molden")
  
#end

