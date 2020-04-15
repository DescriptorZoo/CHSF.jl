using CHSF
using Test, LinearAlgebra, BenchmarkTools
using JuLIP, JuLIP.Testing
using CHSF: chsf

@testset "CHSF.jl" begin
    include("test.jl")
end

