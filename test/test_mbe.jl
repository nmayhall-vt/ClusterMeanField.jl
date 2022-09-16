
using ClusterMeanField  
using LinearAlgebra
using Printf
using Test
using NPZ
using InCoreIntegrals
using ActiveSpaceSolvers

@testset "MBE" begin
    atoms = []
    push!(atoms,Atom(1,"He",[0,0,0]))
    push!(atoms,Atom(2,"He",[2,.1,0]))
    push!(atoms,Atom(3,"He",[4,0,0]))
    push!(atoms,Atom(4,"He",[6,0,0]))
    basis = "6-31g"

    mol     = Molecule(0,1,atoms,basis)
    mf = ClusterMeanField.pyscf_do_scf(mol)
    nbas = size(mf.mo_coeff)[1]
    ints = ClusterMeanField.pyscf_build_ints(mol,mf.mo_coeff, zeros(nbas,nbas));
    e_fci, d1a_fci, d1b_fci,d2_fci = ClusterMeanField.pyscf_fci(ints,6,6)

    println()
    @printf(" RHF Energy: %12.8f\n", mf.e_tot)
    @printf(" FCI Energy: %12.8f\n", e_fci+ints.h0)
    clusters    = [(1,),(2,),(3,),(4,),(5,6,7,8)]
    init_fspace = [(1,1),(1,1),(1,1),(1,1),(0,0)]

    clusters = [Cluster(i,collect(clusters[i])) for i = 1:length(clusters)]
    display(clusters)

    C = mf.mo_coeff
    P = mf.make_rdm1()
    S = mf.get_ovlp()
   
    P = C'*S*P*S*C
    Pa = P*.5
    Pb = P*.5
    out_hf = ClusterMeanField.Increment(clusters, Pa, Pb)
    ehf = compute_energy(ints, out_hf)
    @test isapprox(ehf, mf.e_tot, atol=1e-9)
    f = ClusterMeanField.build_fockian(ints, out_hf)
    tmp = mf.get_fock()
    tmp = mf.mo_coeff'*tmp*mf.mo_coeff

    @test isapprox(norm(f[1] - tmp), 0, atol=1e-12)

    out, increments = ClusterMeanField.gamma_mbe(3, ints, clusters, init_fspace, Pa, Pb, verbose=0)
    display(out.E[1])
    @test isapprox(out.E[1], -11.468550564963303, atol=1e-12)
    e = compute_energy(ints, out)
    display(e)
    @test isapprox(e, -11.470870327910571, atol=1e-12)

    ClusterMeanField.update_2rdm_with_cumulant!(out)
    e = compute_energy(ints, out)
    @test isapprox(e, -11.468585684352316, atol=1e-12)
    display(e)

    out, increments = ClusterMeanField.gamma_mbe(5, ints, clusters, init_fspace, Pa, Pb, verbose=0)
    err = norm(out.Da - ClusterMeanField.integrate_rdm2(out.Daa))
    @test isapprox(err, 0.0, atol=1e-12)
end


