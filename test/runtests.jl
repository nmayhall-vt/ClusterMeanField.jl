using ClusterMeanField
using Test

@testset "ClusterMeanField.jl" begin
    include("test_cmf.jl")
    include("test_savg.jl")
end
