using ClusterMeanField


"""
    build_orbital_gradient(ints, rdm1, rdm2)

Build the full orbital rotation hessian
g_{pq} = <[H,p'q-q'p]> for p<q

# Arguments
- `ints`: Integrals
- `rdm1`: Spin summed 1RDM, NxN
- `rdm2`: Spin summed 2RDM, NxNxNxN
"""
function build_orbital_gradient(ints::InCoreInts, rdm1::Array{T,2}, rdm2::Array{T,4}; verbose=0) where T
    verbose == 0 || println(" In build_orbital_gradient")
    
    N = n_orb(ints)
    G = zeros(N,N)
    g = zeros(N*(N-1)÷2)
   
    G = ints.h1*rdm1 
    V = ints.h2

    @tensor begin
        G[p,q] +=  V[r,p,s,t] * rdm2[r,q,s,t]
    end
    return pack_gradient(2*(G-G'),N)
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
function build_orbital_hessian(ints::InCoreInts, rdm1, rdm2; verbose=0)
    verbose == 0 || println(" In build_orbital_hessian")
end
