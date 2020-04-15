module CHSF

using JuLIP
using LinearAlgebra

export chsf, chsf_desc

chsfPI = 3.14159265358979

function chebyshev(x, n)
    T = zeros((n+1, size(x,1)))
    T[1,:] = ones(size(x,1))
    if n > 0 
        T[2,:] = x 
        for i=3:n+1
            T[i,:] = 2.0 .* x .* T[i-1,:] .- T[i-2,:]
        end
    end 
    return T   
end

function cutoff_func(dij, Rc)
    fc = zeros(size(dij))
    dij_in_Rc = dij .< Rc
    fc[dij_in_Rc] = 0.5 .* ( cos.(dij[dij_in_Rc] .* chsfPI/Rc) .+ 1.0)
    return fc  
end

function c_RDF(dj, n, Rc)
    wtj = 1 # Single specie case!
    fc_ij = cutoff_func(dj, Rc) 
    scaled_r = (2.0 .* dj .- Rc) ./ Rc
    phi_ij = chebyshev(scaled_r, n-1)
    return phi_ij * fc_ij .* wtj 
end

function c_ADF(Rj, dj, l, Rc)
    wtj = 1 # Single specie case!
    wtk = 1 # Single specie case!
    phi_Rc = chsfPI
    counts = collect(1:size(Rj,1))
    ### This part is very terrible in Julia. 
    # So far no other way to do this simpler.
    Rij = collect(Iterators.flatten([[Rj[j] for k in counts] for j in counts]))
    Rik = collect(Iterators.flatten([[Rj[k] for k in counts] for j in counts]))
    Dij = collect(Iterators.flatten([[dj[j] for k in counts] for j in counts]))
    Dik = collect(Iterators.flatten([[dj[k] for k in counts] for j in counts]))
    ###
    cos_theta_ijk = [Rik[i,:]' * Rij[i,:] for i=1:size(Rij,1)] ./ (Dij .* Dik)
    scaled_theta = (2.0 .* cos_theta_ijk .- phi_Rc) ./ phi_Rc
    phi_ijk = chebyshev(scaled_theta, l-1)
    fc_ij = cutoff_func(Dij, Rc) .* wtj
    fc_ik = cutoff_func(Dik, Rc) .* wtk
    jkmask = collect(Iterators.flatten([[k>=(j+1) for k in counts] for j in counts]))
    return phi_ijk[:,jkmask] * (fc_ij[jkmask] .* fc_ik[jkmask])
end

function chsf_desc(Rs, Rc; n=nothing, l=nothing)
    ds = norm.(Rs)
    descriptor = []
    if n != nothing
        if n>-1
            # Concatenate radial basis functions
            descriptor = vcat(descriptor,c_RDF(ds, n+1, Rc))
        end
    end
    if l != nothing
        if l>-1
            # Concatenate angular basis functions
            descriptor = vcat(descriptor,c_ADF(Rs, ds, l+1, Rc))
        end
    end
    return descriptor
end

function chsf(at, Rc; n=nothing, l=nothing)
    representation = []
    ni = []
    nj = []
    nR = []
    nd = []
    for (i, j, R) in pairs(neighbourlist(at, Rc))
        push!(ni, i)
        push!(nj, j)
        push!(nR, R)
        push!(nd, norm(R))
    end
    for i=1:length(at)
        Rs = nR[(ni .== i)]
        push!(representation,chsf_desc(Rs, Rc, n=n, l=l))
    end
    return representation
end

end # module
