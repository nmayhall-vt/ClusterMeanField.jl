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
    push!(atoms,Atom(1, "O", [-0.98166,  0.48113,-0.00901]))
    push!(atoms,Atom(2, "H", [ 0.00766,  0.44112, 0.00636]))
    push!(atoms,Atom(3, "H", [-1.27103, -0.40295, 0.33051]))
    basis = "6-31g"

    mol     = Molecule(0,1,atoms,basis)
    mf = ClusterMeanField.pyscf_do_scf(mol)
    nbas = size(mf.mo_coeff)[1]
    ints = ClusterMeanField.pyscf_build_ints(mol,mf.mo_coeff, zeros(nbas,nbas));
    #e_fci, d1a_fci, d1b_fci,d2_fci = ClusterMeanField.pyscf_fci(ints,5,5)
    e_fci = -76.12171273 - ints.h0
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
    
    clusters    = [(1,),(2,3,4,5),(6,),(7,),(8,),(9,),(10,),(11,),(12,),(13,)]
    init_fspace = [(1,1),(4,4),(0,0),(0,0),(0,0),(0,0),(0,0),(0,0),(0,0),(0,0)]
    
    clusters    = [(1,),(2,3,4,5),(6,7,8,9),(10,11),(12,13,)]
    init_fspace = [(1,1),(4,4),(0,0),(0,0),(0,0)]
    
    clusters    = [(1,),(2,3,4,5),(6,),(7,),(8,),(9,),(10,),(11,),(12,),(13,)]
    init_fspace = [(1,1),(4,4),(0,0),(0,0),(0,0),(0,0),(0,0),(0,0),(0,0),(0,0)]
    
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

    ev = zeros(4,5)
    if true 
        for n in 1:size(ev,2)
            out, increments = ClusterMeanField.gamma_mbe(n, ints, clusters, init_fspace, Pa, Pb)
            @printf(" EFCI: %12.8f\n", e_fci+ints.h0)

            out2 = deepcopy(out)
            out3 = deepcopy(out)

            ClusterMeanField.update_1rdm_with_2rdm!(out2)
            ClusterMeanField.update_2rdm_with_cumulant!(out3)

            e1 = out.E[1]
            e2 = compute_energy(ints, out)
            e3 = compute_energy(ints, out2)
            e4 = compute_energy(ints, out3)
            @printf(" Energy with E expansion:        %12.8f\n", e1)
            @printf(" Energy with 1/2rdm expansion:   %12.8f\n", e2)
            @printf(" Energy with 2rdm expansion:     %12.8f\n", e3)
            @printf(" Energy with cumulant expansion: %12.8f\n", e4)
            ev[1,n] = e1
            ev[2,n] = e2
            ev[3,n] = e3
            ev[4,n] = e4
        end
    end

    display(e_fci)
        
    #out, increments = ClusterMeanField.gamma_mbe(3, ints, clusters, init_fspace, Pa, Pb)

    println(e_fci+ints.h0)
    return ints, out, increments, ev
end

ints, out, increments, ev = run();
