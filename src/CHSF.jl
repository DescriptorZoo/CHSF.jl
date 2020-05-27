module CHSF

using JuLIP
using LinearAlgebra

export chsf, chsf2, chsf3, chsf_desc, chsf_desc2, chsf_desc3

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

function c_ADF2(Rj, dj, l, Rc; m=1)
    if m<1
        m=1
    end
    wtj = 1 # Single specie case!
    wtk = 1 # Single specie case!
    phi_Rc = chsfPI^m
    counts = collect(1:size(Rj,1))
    ### This part is very terrible in Julia. 
    # So far no other way to do this simpler.
    Rij = collect(Iterators.flatten([[Rj[j] for k in counts] for j in counts]))
    Rik = collect(Iterators.flatten([[Rj[k] for k in counts] for j in counts]))
    Dij = collect(Iterators.flatten([[dj[j] for k in counts] for j in counts]))
    Dik = collect(Iterators.flatten([[dj[k] for k in counts] for j in counts]))
    ###
    cos_theta_ijk = [Rik[i,:]' * Rij[i,:] for i=1:size(Rij,1)] ./ (Dij .* Dik)
    scaled_theta = (2.0 .* cos_theta_ijk.^(2*m) .- phi_Rc) ./ phi_Rc
    phi_ijk = chebyshev(scaled_theta, l-1)
    fc_ij = cutoff_func(Dij, Rc) .* wtj
    fc_ik = cutoff_func(Dik, Rc) .* wtk
    jkmask = collect(Iterators.flatten([[k>=(j+1) for k in counts] for j in counts]))
    return phi_ijk[:,jkmask] * (fc_ij[jkmask] .* fc_ik[jkmask])
end

function c_RADF2(Rj, dj, n, l, Rc; m=1)
    wtj = 1 # Single specie case!
    wtk = 1 # Single specie case!
    if m<1
        m=1
    end
    phi_Rc = chsfPI.^(2*m)
    counts = collect(1:size(Rj,1))
    ### So far I couldn't find another way to do this simpler.
    Rij = collect(Iterators.flatten([[Rj[j] for k in counts] for j in counts]))
    Rik = collect(Iterators.flatten([[Rj[k] for k in counts] for j in counts]))
    Dij = collect(Iterators.flatten([[dj[j] for k in counts] for j in counts]))
    Dik = collect(Iterators.flatten([[dj[k] for k in counts] for j in counts]))
    ###
    cos_theta_ijk = [Rik[i,:]' * Rij[i,:] for i=1:size(Rij,1)] ./ (Dij .* Dik)
    scaled_theta = (2.0 .* cos_theta_ijk.^(2*m) .- phi_Rc) ./ phi_Rc
    phi_ijk = chebyshev(scaled_theta, l-1)
    fc_ij = cutoff_func(Dij, Rc) .* wtj
    fc_ik = cutoff_func(Dik, Rc) .* wtk
    fc_j = cutoff_func(dj, Rc) .* wtj
    scaled_r = (2.0 .* dj.^(2*m) .- Rc) ./ Rc
    phi_ij = chebyshev(scaled_r, n-1)
    jkmask = collect(Iterators.flatten([[k>=(j+1) for k in counts] for j in counts]))
    return collect(Iterators.flatten((phi_ijk[:,jkmask] * (fc_ij[jkmask] .* fc_ik[jkmask])) .* transpose(phi_ij * fc_j)))
end

function c_RADF(Rj, dj, n, l, Rc; g=false)
    wtj = 1 # Single specie case!
    wtk = 1 # Single specie case!
    phi_Rc = chsfPI
    counts = collect(1:size(Rj,1))
    ### So far I couldn't find another way to do this simpler.
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
    scaled_r_ij = (2.0 .* Dij .- Rc) ./ Rc
    scaled_r_ik = (2.0 .* Dik .- Rc) ./ Rc
    phi_ij = chebyshev(scaled_r_ij, n-1)
    phi_ik = chebyshev(scaled_r_ik, n-1)
    jkmask = collect(Iterators.flatten([[k>=(j+1) for k in counts] for j in counts]))
    if g
        return collect(Iterators.flatten((phi_ijk[:,jkmask] * (exp.(Dij[jkmask].^2) .* fc_ij[jkmask] .* fc_ik[jkmask])) .* transpose(phi_ij[:,jkmask] * fc_ij[jkmask] .* phi_ik[:,jkmask] * fc_ik[jkmask])))
    else
        return collect(Iterators.flatten((phi_ijk[:,jkmask] * (fc_ij[jkmask] .* fc_ik[jkmask])) .* transpose(phi_ij[:,jkmask] * fc_ij[jkmask] .* phi_ik[:,jkmask] * fc_ik[jkmask])))
    end
end

function chsf_desc2(Rs, Rc; n=nothing, l=nothing, g=false)
    ds = norm.(Rs)
    descriptor = []
    if n == nothing
        n = 0
    end
    if l == nothing
        l = 0
    end
    descriptor = vcat(descriptor,c_RADF(Rs, ds, n+1, l+1, Rc, g=g))
    return descriptor
end

function chsf_desc3(Rs, Rc; n=nothing, l=nothing, m=nothing)
    if m==nothing
        m=2
    end
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
            descriptor = vcat(descriptor,c_ADF2(Rs, ds, l+1, Rc, m=m))
        end
    end
    return descriptor
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

function chsf2(at, Rc; n=nothing, l=nothing, g=false)
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
        push!(representation,chsf_desc2(Rs, Rc, n=n, l=l, g=g))
    end
    return representation
end

function chsf3(at, Rc; n=nothing, l=nothing, m=nothing)
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
        push!(representation,chsf_desc3(Rs, Rc, n=n, l=l, m=m))
    end
    return representation
end

end # module
