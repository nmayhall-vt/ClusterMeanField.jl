using ClusterMeanField


"""
    build_orbital_gradient(ints, rdm1, rdm2)

Build the full orbital rotation hessian
g_{pq} = <[H,p'q-q'p]>

# Arguments
- `ints`: Integrals
- `rdm1`: Spin summed 1RDM
- `rdm2`: Spin summed 2RDM
"""
function build_orbital_gradient(ints::InCoreInts, rdm1, rdm2)
end

"""
    build_orbital_hessian(ints, rdm1, rdm2)

Build the full orbital rotation hessian
H_{pq,rs} = <[[H,p'q-q'p], r's-s'r]>

# Arguments
- `ints`: Integrals
- `rdm1`: Spin summed 1RDM
- `rdm2`: Spin summed 2RDM
"""
function build_orbital_hessian(ints::InCoreInts, rdm1, rdm2)
end
