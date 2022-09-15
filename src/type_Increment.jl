using ClusterMeanField

struct Increment{T}
    #term::NTuple{N,Int}
    clusters::Vector{Cluster}
    E::Array{T,1}
    Pa::Array{T,2}
    Pb::Array{T,2}
    Gaa::Array{T,4}
    Gab::Array{T,4}
    Gbb::Array{T,4}
end

"""
    Increment(n::Int; T=Float64)
"""
function Increment(; n=1, T=Float64)
    return Increment{T}([Cluster(0, [i for i in 1:n])], [0.0], zeros(n,n), zeros(n,n), zeros(n,n,n,n), zeros(n,n,n,n), zeros(n,n,n,n))
end

"""
    Increment(c::Cluster; T=Float64)
"""
function Increment(clusters::Vector{Cluster}; T=Float64)
    n = 0
    for ci in clusters 
        n += length(ci)
    end
    return Increment{T}(clusters, [0.0], zeros(n,n), zeros(n,n), zeros(n,n,n,n), zeros(n,n,n,n), zeros(n,n,n,n))
end

"""
    Increment(da::Array{T,2}, db::Array{T,2})  where {T}
"""
function Increment(clusters::Vector{Cluster}, da::Array{T,2}, db::Array{T,2})  where {T}
    n = size(da,1)
    n == size(da,2) || throw(DimensionMismatch)
    n == size(db,1) || throw(DimensionMismatch)
    n == size(db,2) || throw(DimensionMismatch)
    incr = Increment{T}(clusters, [0.0], da, db, zeros(n,n,n,n), zeros(n,n,n,n), zeros(n,n,n,n))
    Gaa = incr.Gaa
    Gab = incr.Gab
    Gbb = incr.Gbb
    @tensor begin
        Gaa[p,q,r,s] += da[p,q] * da[r,s]
        Gaa[p,q,r,s] -= da[p,s] * da[r,q]

        Gbb[p,q,r,s] += db[p,q] * db[r,s]
        Gbb[p,q,r,s] -= db[p,s] * db[r,q]

        Gab[p,q,r,s] += da[p,q] * db[r,s]
    end
    incr.Gaa .= Gaa
    incr.Gab .= Gab
    incr.Gbb .= Gbb
    return incr
end
function Base.display(incr::Increment)
    @printf(" :Increment:   ")
    [@printf("%3i",i.idx) for i in incr.clusters]
    println()
    @printf("   E:          %12.8f\n",incr.E[1])
    @printf("   tr(Pa):     %12.8f\n",tr(incr.Pa))
    @printf("   tr(Pb):     %12.8f\n",tr(incr.Pb))
    @printf("   tr(Gaa):    %12.8f\n",tr(incr.Gaa)*.5)
    @printf("   tr(Gab):    %12.8f\n",tr(incr.Gab))
    @printf("   tr(Gbb):    %12.8f\n",tr(incr.Gbb)*.5)
end

function InCoreIntegrals.compute_energy(ints::InCoreInts, incr::Increment)
    e = ints.h0
    e += sum(ints.h1 .* incr.Pa)
    e += sum(ints.h1 .* incr.Pb)
    e += .5*sum(ints.h2 .* incr.Gaa)
    e += .5*sum(ints.h2 .* incr.Gbb)
    e += sum(ints.h2 .* incr.Gab)
    return e
end

"""
"""
function Base.:-(i1::Increment, i2::Increment)
    orbs1 = union((c.orb_list for c in i1.clusters)...)
    orbs2 = union((c.orb_list for c in i2.clusters)...)
    
    ind1 = findall(x->x in orbs2, orbs1) # coinciding indices in i1
    ind2 = findall(x->x in orbs1, orbs2) # coinciding indices in i2
   
    length(ind1) == length(ind2) || error("huh?")

    #display(ind1)
    #display(ind2)
    out = deepcopy(i1)
    out.E .= i1.E .- i2.E
    out.Pa[ind1, ind1] .-= i2.Pa[ind2, ind2]
    out.Pb[ind1, ind1] .-= i2.Pb[ind2, ind2]
    out.Gaa[ind1, ind1, ind1, ind1] .-= i2.Gaa[ind2, ind2, ind2, ind2]
    out.Gab[ind1, ind1, ind1, ind1] .-= i2.Gab[ind2, ind2, ind2, ind2]
    out.Gbb[ind1, ind1, ind1, ind1] .-= i2.Gbb[ind2, ind2, ind2, ind2]

    return out 
end

"""
"""
function Base.:+(i1::Increment, i2::Increment)
    orbs1 = union((c.orb_list for c in i1.clusters)...)
    orbs2 = union((c.orb_list for c in i2.clusters)...)
    
    ind1 = findall(x->x in orbs2, orbs1) # coinciding indices in i1
    ind2 = findall(x->x in orbs1, orbs2) # coinciding indices in i2
   
    length(ind1) == length(ind2) || error("huh?")

    #display(ind1)
    #display(ind2)
    out = deepcopy(i1)
    out.E .+= i2.E
    out.Pa[ind1, ind1] .+= i2.Pa[ind2, ind2]
    out.Pb[ind1, ind1] .+= i2.Pb[ind2, ind2]
    out.Gaa[ind1, ind1, ind1, ind1] .+= i2.Gaa[ind2, ind2, ind2, ind2]
    out.Gab[ind1, ind1, ind1, ind1] .+= i2.Gab[ind2, ind2, ind2, ind2]
    out.Gbb[ind1, ind1, ind1, ind1] .+= i2.Gbb[ind2, ind2, ind2, ind2]

    return out 
end
