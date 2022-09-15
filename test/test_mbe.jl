
using ClusterMeanField  
using LinearAlgebra
using Printf
using Test
using NPZ
using InCoreIntegrals
using ActiveSpaceSolvers

function run2()
    atoms = []
    R = .75
    push!(atoms,Atom(1,"H",[-R, 0, 0]))
    push!(atoms,Atom(2,"H",[ R, 0, 0]))
    push!(atoms,Atom(3,"H",[ 0,-R, 0]))
    push!(atoms,Atom(4,"H",[ 0, R, 0]))
    push!(atoms,Atom(5,"H",[ 0, 0,-R]))
    push!(atoms,Atom(6,"H",[ 0, 0, R]))
    basis = "sto-3g"

    mol     = Molecule(0,1,atoms,basis)
    mf = ClusterMeanField.pyscf_do_scf(mol)
    return mf
end

function run()
# 
    atoms = []
    push!(atoms,Atom(1,"He",[0,0,0]))
    push!(atoms,Atom(2,"He",[1,0,0]))
    push!(atoms,Atom(3,"He",[2,0,0]))
    push!(atoms,Atom(4,"He",[3,0,0]))
    #push!(atoms,Atom(5,"He",[4,0,0]))
    #push!(atoms,Atom(6,"He",[5,0,0]))
    basis = "sto-3g"
    basis = "6-31g"

    mol     = Molecule(0,1,atoms,basis)
    mf = ClusterMeanField.pyscf_do_scf(mol)
    nbas = size(mf.mo_coeff)[1]
    ints = ClusterMeanField.pyscf_build_ints(mol,mf.mo_coeff, zeros(nbas,nbas));
    e_fci, d1a_fci, d1b_fci,d2_fci = ClusterMeanField.pyscf_fci(ints,4,4)

    println()
    @printf(" RHF Energy: %12.8f\n", mf.e_tot)
    @printf(" FCI Energy: %12.8f\n", e_fci+ints.h0)

    ClusterMeanField.pyscf_write_molden(mol,mf.mo_coeff,filename="scf.molden")
    C = mf.mo_coeff
    #rdm_mf = C[:,1:2] * C[:,1:2]'
    #Cl = ClusterMeanField.localize(mf.mo_coeff,"lowdin",mf)
    #ClusterMeanField.pyscf_write_molden(mol,Cl,filename="lowdin.molden")
    #S = ClusterMeanField.get_ovlp(mf)
    #U =  C' * S * Cl
    #println(" Build Integrals")
    #flush(stdout)
    #ints = orbital_rotation(ints,U)
    #println(" done.")
    #flush(stdout)

    #clusters    = [(1,),(2,),(3,),(4,),(5,),(6,),(7:12)]
    #init_fspace = [(1,1),(1,1),(1,1)]
    
    clusters    = [(1,),(2,),(3,),(4,),(5,6,7,8)]
    init_fspace = [(1,1),(1,1),(1,1),(1,1),(0,0)]
    clusters = [Cluster(i,collect(clusters[i])) for i = 1:length(clusters)]
    display(clusters)

    C = mf.mo_coeff
    P = mf.make_rdm1()
    S = mf.get_ovlp()
    #display(tr(C'*S*P*S*C))
    #display(C'*S*P*S*C)
   
    P = C'*S*P*S*C
    Pa = P*.5
    Pb = P*.5
    ClusterMeanField.gamma_mbe(ints, clusters, init_fspace, Pa, Pb)
end

run()
