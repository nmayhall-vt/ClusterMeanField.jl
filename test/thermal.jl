using QCBase  
using RDM  
using ClusterMeanField  
using LinearAlgebra
using Printf
using Test
using NPZ
using InCoreIntegrals
using ActiveSpaceSolvers
using PyCall


atoms = []
push!(atoms,Atom(1,"H",[0,0,0]))
push!(atoms,Atom(2,"H",[1,0,0]))
push!(atoms,Atom(3,"H",[2,0,0]))
push!(atoms,Atom(4,"H",[3,0,0]))
push!(atoms,Atom(5,"H",[4,0,0]))
push!(atoms,Atom(6,"H",[5,0,0]))
basis = "sto-3g"

mol     = Molecule(0,1,atoms,basis)
mf      = ClusterMeanField.pyscf_do_scf(mol)
ints    = ClusterMeanField.pyscf_build_ints(mol,mf.mo_coeff);

norb = n_orb(ints)
n_elec_a = 3
n_elec_b = 3
ansatz = FCIAnsatz(norb, n_elec_a, n_elec_b)
display(ansatz)
    
Hmat = build_H_matrix(ints, ansatz)
    
F = eigen(Hmat)
T = 20000.
k = 3.166811563e-6 # Hartree/Kelvin
β = 1.0/(k*T)

display(β)
rho = F.vectors' * diagm(exp.(-β.*F.values)) * F.vectors
Z = tr(rho)
rho .= rho ./ Z 


