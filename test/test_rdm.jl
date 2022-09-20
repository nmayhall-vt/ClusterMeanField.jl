using ClusterMeanField  
using LinearAlgebra
using Printf
using Test
using NPZ
using InCoreIntegrals
using ActiveSpaceSolvers
using PyCall

@testset "RDM" begin
    h0 = npzread("h6_sto3g/h0.npy")
    h1 = npzread("h6_sto3g/h1.npy")
    h2 = npzread("h6_sto3g/h2.npy")
    
    ints = InCoreInts(h0, h1, h2)

    pyscf = pyimport("pyscf")
    fci = pyimport("pyscf.fci")
    cisolver = pyscf.fci.direct_spin1.FCI()
    cisolver.max_cycle = 100 
    cisolver.conv_tol = 1e-9
    
    norb = n_orb(ints) 
    efci, ci = cisolver.kernel(ints.h1, ints.h2, norb, (3,3), ecore=ints.h0, nroots = 1, verbose=0)
    fci_dim = size(ci,1)*size(ci,2)

    d1,d2 = cisolver.make_rdm12s(ci, norb, (3,3))
    
    d1 = RDM1(d1[1], d1[2])
    d2 = RDM2(d2[1], d2[2], d2[3])

    ssd1 = ssRDM1(d1)
    ssd2 = ssRDM2(d2)

    @printf(" FCI: %12.8f\n",efci)
    @printf(" RDM: %12.8f\n", compute_energy(ints, d1, d2))
    @printf(" RDM: %12.8f\n", compute_energy(ints, ssd1.rdm, ssd2.rdm))

    @test isapprox(efci, compute_energy(ints, d1, d2), atol=1e-12)
    @test isapprox(efci, compute_energy(ints, ssd1, ssd2), atol=1e-12)

    display(norm(d1.a - RDM1(d2).a))
    display(norm(d1.b - RDM1(d2).b))
    @test isapprox(norm(d1.a - RDM1(d2).a),0,atol=1e-14)
    @test isapprox(norm(d1.b - RDM1(d2).b),0,atol=1e-14)

end

