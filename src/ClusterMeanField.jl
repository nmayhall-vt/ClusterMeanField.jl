module ClusterMeanField

using QCBase
using RDM

using LinearAlgebra
using Random
using Optim
using TensorOperations
using InCoreIntegrals
using Printf
using ActiveSpaceSolvers

export form_1rdm_dressed_ints
export cmf_ci
export cmf_oo

export pyscf_do_scf
export make_pyscf_mole
export pyscf_write_molden
export pyscf_build_1e
export pyscf_build_eri
export pyscf_get_jk
export pyscf_build_ints
export pyscf_fci
export get_nuclear_rep
export localize
export get_ovlp

#Base.convert(::Type{Vector{MOCluster}}, in::Vector{MOCluster{N}}) where {N} = return Vector{MOCluster}(in)


include("PyscfFunctions.jl")
include("incore_cmf.jl")
include("direct_cmf.jl")

end
