using QCBase
using ClusterMeanField
using ActiveSpaceSolvers
using RDM
using InCoreIntegrals
using PyCall
using JLD2

function run_cmf()

    pyscf = pyimport("pyscf");
    fcidump = pyimport("pyscf.tools.fcidump");
    ctx = fcidump.read("/Users/nicole/My Drive/code/FermiCG-data/excited_paper/p1/fcidump_4mer");
    h = ctx["H1"];
    g = ctx["H2"];
    ecore = ctx["ECORE"];
    g = pyscf.ao2mo.restore("1", g, size(h,2))
    ints = InCoreInts(ecore,h,g)

    clusters_in    = [(1:6),(7:12),(13:18),(19:24)]
    n_clusters = 4
    cluster_list = [collect(1:6), collect(7:12), collect(13:18), collect(19:24)]
    clusters = [MOCluster(i,collect(cluster_list[i])) for i = 1:length(cluster_list)]
    init_fspace = [ (3,3) for i in 1:n_clusters]
    #init_cluster_ansatz = [FCIAnsatz(6, 3, 3), FCIAnsatz(6, 3, 3), FCIAnsatz(6, 3, 3), FCIAnsatz(6, 3, 3)]
    #init_cluster_ansatz = [RASCIAnsatz(6, 3, 3, (2,2,2)), RASCIAnsatz(6,3,3,(2,2,2)), RASCIAnsatz(6,3,3,(2,2,2)), RASCIAnsatz(6,3,3,(2,2,2))]
    init_cluster_ansatz = [RASCIAnsatz(6, 3, 3, (2,2,2), 1, 1), RASCIAnsatz(6,3,3,(2,2,2), 1, 1), RASCIAnsatz(6,3,3,(2,2,2), 1, 1), RASCIAnsatz(6,3,3,(2,2,2), 1, 1)]
    #delta_elec = [1,1,1,1]
    #ansatze = ActiveSpaceSolvers.generate_cluster_fock_ansatze(init_fspace, clusters, init_cluster_ansatz, delta_elec)

    #CMF has default of delta_elec=0 for all clusters
    ansatze = ActiveSpaceSolvers.generate_cluster_fock_ansatze(init_fspace, clusters, init_cluster_ansatz)

    rdm1 = zeros(size(ints.h1))
    #display(rdm1)
    #ints_i = subset(ints, clusters[1], RDM1(rdm1, rdm1))
    #return ints_i, RDM1(rdm1, rdm1)


    #run cmf_oo
    #e_cmf, U_cmf, d1  = ClusterMeanField.cmf_oo_diis(ints, clusters, init_fspace, ansatze, RDM1(rdm1, rdm1), maxiter_oo = 20, verbose=0, diis_start=3);
    #e_cmf, U_cmf, d1 = ClusterMeanField.cmf_oo(ints, clusters, init_fspace, ansatze, RDM1(rdm1, rdm1), max_iter_oo=200, verbose=0, gconv=1e-6, method="bfgs");
    e_cmf, U_cmf, d1  = ClusterMeanField.cmf_oo_diis(ints, clusters, init_fspace, RDM1(rdm1, rdm1), maxiter_oo=4, verbose=0, diis_start=3);
    #return e_cmf, U_cmf, d1
    #cluster_bases = FermiCG.compute_cluster_eigenbasis_spin(ints, clusters, d1, [3,3,3,3], init_fspace, max_roots=20, verbose=1);
    #@save "testing_ansatz.jld2" ints clusters d1 init_fspace rdm1 
    #return ints, clusters, d1
end

