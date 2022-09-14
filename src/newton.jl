using ClusterMeanField


"""
    build_orbital_gradient(ints, rdm1, rdm2)

Build the full orbital rotation hessian
g_{pq} = <[H,p'q-q'p]> for p<q

# Arguments
- `ints`: Integrals
- `rdm1`: Spin summed 1RDM
- `rdm2`: Spin summed 2RDM
"""
function build_orbital_gradient(ints::InCoreInts, rdm1::Array{T,2}, rdm2::Array{T,4}; verbose=0) where T
    verbose == 0 || println(" In build_orbital_gradient")
    
    N = n_orb(ints)
    G = zeros(N,N)
    g = zeros(N*(N-1)÷2)
   
    G = ints.h1*rdm1 
    V = ints.h2

    #println(rdm2[1,2,3,4])
    #println(rdm2[1,4,3,2]*2)
    @tensor begin
        G[p,q] +=  V[r,s,p,t] * rdm2[r,s,t,q]
        #G[p,q] -=  V[r,s,t,p] * rdm2[r,s,t,q]
        #G[p,q] +=  V[r,q,u,t] * rdm2[p,r,t,u]
        #G[p,q] -=  V[q,s,u,t] * rdm2[p,s,t,u]
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
