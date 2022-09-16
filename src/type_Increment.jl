using ClusterMeanField



struct Increment{T}
    #term::NTuple{N,Int}
    clusters::Vector{Cluster}
    # energy 
    E::Array{T,1}
    # 1-rdm 
    Da::Array{T,2}
    Db::Array{T,2}
    # 2-rdm 
    Daa::Array{T,4}
    Dab::Array{T,4}
    Dbb::Array{T,4}
    # cumulants
    Caa::Array{T,4}
    Cab::Array{T,4}
    Cbb::Array{T,4}
end


"""
    Increment(c::Cluster; T=Float64)
"""
function Increment(clusters::Vector{Cluster}; T=Float64)
    n = 0
    for ci in clusters 
        n += length(ci)
    end
    println("n: ", n)
    return Increment{T}(clusters, [0.0], 
                        zeros(n,n), zeros(n,n), 
                        zeros(n,n,n,n), zeros(n,n,n,n), zeros(n,n,n,n),
                        zeros(n,n,n,n), zeros(n,n,n,n), zeros(n,n,n,n))
end

"""
    Increment(da::Array{T,2}, db::Array{T,2})  where {T}
"""
function Increment(clusters::Vector{Cluster}, da::Array{T,2}, db::Array{T,2})  where {T}
    n = size(da,1)
    n == size(da,2) || throw(DimensionMismatch)
    n == size(db,1) || throw(DimensionMismatch)
    n == size(db,2) || throw(DimensionMismatch)
    incr = Increment{T}(clusters, [0.0], da, db, 
                        zeros(n,n,n,n), zeros(n,n,n,n), zeros(n,n,n,n),
                        zeros(n,n,n,n), zeros(n,n,n,n), zeros(n,n,n,n))
    Daa = incr.Daa
    Dab = incr.Dab
    Dbb = incr.Dbb
    @tensor begin
        Daa[p,q,r,s] += da[p,q] * da[r,s]
        Daa[p,q,r,s] -= da[p,s] * da[r,q]

        Dbb[p,q,r,s] += db[p,q] * db[r,s]
        Dbb[p,q,r,s] -= db[p,s] * db[r,q]

        Dab[p,q,r,s] += da[p,q] * db[r,s]
    end
    incr.Daa .= Daa
    incr.Dab .= Dab
    incr.Dbb .= Dbb
    return incr
end
function Base.display(incr::Increment)
    @printf(" :Increment:   ")
    [@printf("%3i",i.idx) for i in incr.clusters]
    println()
    @printf("   E:       %12.8f\n",incr.E[1])
    @printf("   tr(Da):  %12.8f\n",tr(incr.Da)     )
    @printf("   tr(Db):  %12.8f\n",tr(incr.Db)     )
    @printf("   tr(Daa): %12.8f\n",tr(incr.Daa)*.5 )
    @printf("   tr(Dab): %12.8f\n",tr(incr.Dab)    )
    @printf("   tr(Dbb): %12.8f\n",tr(incr.Dbb)*.5 )
    @printf("   tr(Caa): %12.8f   norm(Caa): %12.8f\n",tr(incr.Caa)*.5 ,norm(incr.Caa) )
    @printf("   tr(Cab): %12.8f   norm(Cab): %12.8f\n",tr(incr.Cab)    ,norm(incr.Cab) )
    @printf("   tr(Cbb): %12.8f   norm(Cbb): %12.8f\n",tr(incr.Cbb)*.5 ,norm(incr.Cbb) )
    
#    @printf("   tr(Da):  %12.8f   norm(Da):  %12.8f\n",tr(incr.Da)     ,norm(incr.Da)  )
#    @printf("   tr(Db):  %12.8f   norm(Db):  %12.8f\n",tr(incr.Db)     ,norm(incr.Db)  )
#    @printf("   tr(Daa): %12.8f   norm(Daa): %12.8f\n",tr(incr.Daa)*.5 ,norm(incr.Daa) )
#    @printf("   tr(Dab): %12.8f   norm(Dab): %12.8f\n",tr(incr.Dab)    ,norm(incr.Dab) )
#    @printf("   tr(Dbb): %12.8f   norm(Dbb): %12.8f\n",tr(incr.Dbb)*.5 ,norm(incr.Dbb) )
#    @printf("   tr(Caa): %12.8f   norm(Caa): %12.8f\n",tr(incr.Caa)*.5 ,norm(incr.Caa) )
#    @printf("   tr(Cab): %12.8f   norm(Cab): %12.8f\n",tr(incr.Cab)    ,norm(incr.Cab) )
#    @printf("   tr(Cbb): %12.8f   norm(Cbb): %12.8f\n",tr(incr.Cbb)*.5 ,norm(incr.Cbb) )
end

function InCoreIntegrals.compute_energy(ints::InCoreInts, incr::Increment)
    e = ints.h0
    e += sum(ints.h1 .* incr.Da)
    e += sum(ints.h1 .* incr.Db)
    e += .5*sum(ints.h2 .* incr.Daa)
    e += .5*sum(ints.h2 .* incr.Dbb)
    e += sum(ints.h2 .* incr.Dab)
    return e
end

"""
    update_2rdm_with_cumulant!(incr::Increment)
"""
function update_2rdm_with_cumulant!(incr::Increment)
    (aa,ab,bb) = build_2rdm((incr.Da, incr.Db))
    incr.Daa .= incr.Caa .+ aa 
    incr.Dab .= incr.Cab .+ ab 
    incr.Dbb .= incr.Cbb .+ bb 
    return 
end

function update_1rdm_with_2rdm!(incr::Increment)
    incr.Da .= integrate_rdm2(incr.Daa) 
    incr.Db .= integrate_rdm2(incr.Dbb) 
    return 
end

"""
    add!(i1::Increment, i2::Increment)

Add `i2` to `i1`, modifying `i1`.
"""
function add!(i1::Increment, i2::Increment)
    orbs1 = union((c.orb_list for c in i1.clusters)...)
    orbs2 = union((c.orb_list for c in i2.clusters)...)
    
    ind1 = findall(x->x in orbs2, orbs1) # coinciding indices in i1
    ind2 = findall(x->x in orbs1, orbs2) # coinciding indices in i2
   
    length(ind1) == length(ind2) || error("huh?")

    i1.E .+= i2.E
    i1.Da[ind1, ind1] .+= i2.Da[ind2, ind2]
    i1.Db[ind1, ind1] .+= i2.Db[ind2, ind2]
    i1.Daa[ind1, ind1, ind1, ind1] .+= i2.Daa[ind2, ind2, ind2, ind2]
    i1.Dab[ind1, ind1, ind1, ind1] .+= i2.Dab[ind2, ind2, ind2, ind2]
    i1.Dbb[ind1, ind1, ind1, ind1] .+= i2.Dbb[ind2, ind2, ind2, ind2]
    i1.Caa[ind1, ind1, ind1, ind1] .+= i2.Caa[ind2, ind2, ind2, ind2]
    i1.Cab[ind1, ind1, ind1, ind1] .+= i2.Cab[ind2, ind2, ind2, ind2]
    i1.Cbb[ind1, ind1, ind1, ind1] .+= i2.Cbb[ind2, ind2, ind2, ind2]
    return
end

"""
    subtract!(i1::Increment, i2::Increment)

Subtract `i2` from `i1`, modifying `i1`.
"""
function subtract!(i1::Increment, i2::Increment)
    orbs1 = union((c.orb_list for c in i1.clusters)...)
    orbs2 = union((c.orb_list for c in i2.clusters)...)
    
    ind1 = findall(x->x in orbs2, orbs1) # coinciding indices in i1
    ind2 = findall(x->x in orbs1, orbs2) # coinciding indices in i2
   
    length(ind1) == length(ind2) || error("huh?")

    i1.E .-= i2.E
    i1.Da[ind1, ind1] .-= i2.Da[ind2, ind2]
    i1.Db[ind1, ind1] .-= i2.Db[ind2, ind2]
    i1.Daa[ind1, ind1, ind1, ind1] .-= i2.Daa[ind2, ind2, ind2, ind2]
    i1.Dab[ind1, ind1, ind1, ind1] .-= i2.Dab[ind2, ind2, ind2, ind2]
    i1.Dbb[ind1, ind1, ind1, ind1] .-= i2.Dbb[ind2, ind2, ind2, ind2]
    i1.Caa[ind1, ind1, ind1, ind1] .-= i2.Caa[ind2, ind2, ind2, ind2]
    i1.Cab[ind1, ind1, ind1, ind1] .-= i2.Cab[ind2, ind2, ind2, ind2]
    i1.Cbb[ind1, ind1, ind1, ind1] .-= i2.Cbb[ind2, ind2, ind2, ind2]

    return
end
"""
"""
function Base.:-(i1::Increment, i2::Increment)
    out = deepcopy(i1)
    subtract!(out,i2)
    return out 
end

"""
"""
function Base.:+(i1::Increment, i2::Increment)
    out = deepcopy(i1)
    add!(out,i2)
    return out 
end
