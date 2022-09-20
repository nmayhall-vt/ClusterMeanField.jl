using ClusterMeanField

struct RDM1{T} <: AbstractArray{T,2} 
    a::Array{T,2}
    b::Array{T,2}
end
struct RDM2{T} <: AbstractArray{T,4} 
    aa::Array{T,4}
    ab::Array{T,4}
    bb::Array{T,4}
end
struct Cumulant2{T} <: AbstractArray{T,4} 
    aa::Array{T,4}
    ab::Array{T,4}
    bb::Array{T,4}
end

# spin-summed RDMs
struct ssRDM1{T} <: AbstractArray{T,2} 
    rdm::Array{T,2}
end
struct ssRDM2{T} <: AbstractArray{T,4} 
    rdm::Array{T,4}
end


Base.size(r::RDM1) = return size(r.a)
Base.size(r::RDM2) = return size(r.aa)


function ssRDM1(rdm::RDM1{T}) where T
    return ssRDM1{T}(rdm.a .+ rdm.b)
end
function ssRDM2(rdm::RDM2{T}) where T
    return ssRDM2{T}(rdm.aa .+ rdm.bb .+ 2 .* rdm.ab)
end

function RDM2(rdm::RDM1{T}) where T
    no = n_orb(rdm) 
    Daa = zeros(no, no, no, no)
    Dab = zeros(no, no, no, no)
    Dbb = zeros(no, no, no, no)
    for p in 1:no, q in 1:no, r in 1:no, s in 1:no
        Daa[p,q,r,s] = rdm.a[p,q] * rdm.a[r,s] - rdm.a[p,s] * rdm.a[r,q]
        Dbb[p,q,r,s] = rdm.b[p,q] * rdm.b[r,s] - rdm.b[p,s] * rdm.b[r,q]
        Dab[p,q,r,s] = rdm.a[p,q] * rdm.b[r,s]
    end
    return ssRDM2{T}(Daa, Dab, Dbb)
end

n_orb(r::RDM1) = size(r.a,1)
n_orb(r::RDM2) = size(r.aa,1)
n_orb(c::Cumulant2) = size(c.a,1)

n_orb(r::ssRDM1) = size(r.rdm,1)
n_orb(r::ssRDM2) = size(r.rdm,1)

"""
    compute_energy(ints::InCoreInts{T}, d1::ssRDM1{T}, d2::ssRDM2{T}) where T

Return energy defined by spin-summed  `rdm1` and `rdm2`.
# Arguments
- `ints`: InCoreInts object
- `d1`:   1 particle reduced density matrix
- `d2`:   2 particle reduced density matrix
"""
function InCoreIntegrals.compute_energy(ints::InCoreInts{T}, d1::ssRDM1{T}, d2::ssRDM2{T}) where T
#={{{=#
    length(d1.rdm) == length(ints.h1) || throw(DimensionMismatch)

    no = n_orb(d1)
    e = ints.h0
    
    for p in 1:no, q in 1:no
        e += ints.h1[p,q] * d1.rdm[p,q]
    end
    
    for p in 1:no, q in 1:no, r in 1:no, s in 1:no
        e += .5 * ints.h2[p,q,r,s] * d2.rdm[p,q,r,s]
    end
    
    return e
end
#=}}}=#

"""
    compute_energy(ints::InCoreInts{T}, d1::RDM1{T}, d2::RDM2{T}) where T

Return energy defined by `rdm1` and `rdm2`.
# Arguments
- `ints`: InCoreInts object
- `d1`:   1 particle reduced density matrix
- `d2`:   2 particle reduced density matrix
"""
function InCoreIntegrals.compute_energy(ints::InCoreInts{T}, d1::RDM1{T}, d2::RDM2{T}) where T
#={{{=#
    length(d1.a) == length(ints.h1) || throw(DimensionMismatch)
    length(d1.b) == length(ints.h1) || throw(DimensionMismatch)

    no = n_orb(d1)
    e = ints.h0
    
    for p in 1:no, q in 1:no
        e += ints.h1[p,q] * (d1.a[p,q] + d1.b[p,q])
    end
    
    for p in 1:no, q in 1:no, r in 1:no, s in 1:no
        e += .5 * ints.h2[p,q,r,s] * d2.aa[p,q,r,s]
        e += .5 * ints.h2[p,q,r,s] * d2.bb[p,q,r,s]
        e +=      ints.h2[p,q,r,s] * d2.ab[p,q,r,s]
    end
    
    return e
end
#=}}}=#

"""
    compute_energy(ints::InCoreInts{T}, rdm1::RDM1{T}) where T

Return energy defined by `rdm1`.
# Arguments
- `ints`: InCoreInts object
- `rdm1`: 1 particle reduced density matrix
"""
function InCoreIntegrals.compute_energy(ints::InCoreInts{T}, rdm1::RDM1{T}) where T
#={{{=#
    length(rdm1.a) == length(ints.h1) || throw(DimensionMismatch)
    length(rdm1.b) == length(ints.h1) || throw(DimensionMismatch)

    e = ints.h0
    
    for p in 1:no, q in 1:no
        e += ints.h1[p,q] * (rdm1.a[p,q] + rdm.b[p,q])
    end
    
    for p in 1:no, q in 1:no, r in 1:no, s in 1:no
        e += .5 * ints.h2[p,q,r,s] * rdm1.a[p,q] * rdm1.a[r,s]
        e -= .5 * ints.h2[p,q,r,s] * rdm1.a[p,s] * rdm1.a[r,q]
        
        e += .5 * ints.h2[p,q,r,s] * rdm1.b[p,q] * rdm1.b[r,s]
        e -= .5 * ints.h2[p,q,r,s] * rdm1.b[p,s] * rdm1.b[r,q]
        
        e += ints.h2[p,q,r,s] * rdm1.a[p,q] * rdm1.b[r,s]
    end
    
    return e
end
#=}}}=#


"""
    compute_fock(ints::InCoreInts, rdm1::RDM1)

Compute Fock Matrix
"""
function compute_fock(ints::InCoreInts, rdm1::RDM1)
#={{{=#
    fa = deepcopy(ints.h1)
    fb = deepcopy(ints.h1)
    @tensor begin
        #a
        fa[r,s] += 0.5 * ints.h2[p,q,r,s] * rdm1.a[p,q] 
        fa[r,s] -= 0.5 * ints.h2[p,r,q,s] * rdm1.a[p,q]
        fa[r,s] += 0.5 * ints.h2[p,q,r,s] * rdm1.b[p,q] 
        
        #b
        fb[r,s] += 0.5 * ints.h2[p,q,r,s] * rdm1.b[p,q] 
        fb[r,s] -= 0.5 * ints.h2[p,r,q,s] * rdm1.b[p,q]
        fb[r,s] += 0.5 * ints.h2[p,q,r,s] * rdm1.a[p,q]
        
    end
    return (fa,fb) 
end
#=}}}=#


"""
    RDM1(rdm::RDM2)

Integrate a 2rdm to get the 1rdm.
We assume that d is stored, d[1,1,2,2]
such that <p'q'rs> is D[p,s,q,r]
Also, we will agree with pyscf, and use the following normalization:
tr(D) = N(N-1)
"""
function RDM1(d2::RDM2{T}) where T
    n = n_orb(d2) 
    d1a = zeros(n,n)
    d1b = zeros(n,n)
    for p in 1:n
        for q in 1:n
            for r in 1:n
                d1a[p,q] += d2.aa[p,q,r,r]
                d1b[p,q] += d2.bb[p,q,r,r]
            end
        end
    end
    ca = tr(d1a)
    cb = tr(d1b)
    Na = (1 + sqrt(1+4*ca) )/2
    Nb = (1 + sqrt(1+4*cb) )/2
    return RDM1{T}(d1a ./ (Na-1), d1b ./ (Nb-1))
end

Base.:-(da::RDM1, db::RDM1) = return RDM1(da.a.-db.a, da.b.-db.b) 
Base.:+(da::RDM1, db::RDM1) = return RDM1(da.a.+db.a, da.b.+db.b) 
function Base.display(da::RDM1)
    display("RDM1(α)")
    display(da.a)
    display("RDM1(β)")
    display(da.b)
end


