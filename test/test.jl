@testset "CHSF Descriptor" begin


@info("Testing CHSF Descriptor for DC 4x4x4 Si with cutoff=6.5, n=2, l=2")
using CHSF, JuLIP, Test

at = bulk(:Si, cubic=true)
desc = chsf(at, 6.5, n=2, l=2)
#round.(desc[1], digit=7) == chsfpy
chsf_ref = [10.3698237,1.4503467,-8.2118063,51.6882200,-53.0113716,69.8233316] #n=2,l=2 case
chsf_now = vcat(desc[1,:]...)
println("CHSF.jl returns:",chsf_now)
println("Reference:",chsf_ref)
println(@test chsf_now  ≈  chsf_ref)

end
