module CHSF

using JuLIP
using LinearAlgebra

export chsf, chsf_RADF, chsf_desc_RADF, chsf_desc

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

function ch_T(x, n)
    T = zeros(n+1)
    T[1] = 1
    if n > 0 
        T[2] = x 
        for i=3:n+1
            T[i] = 2.0 * x * T[i-1] - T[i-2]
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
    fc_ij = cutoff_func(dj, Rc)
    D = zeros(l)
    for j = 1:length(Rj)-1, k = j+1:length(Rj)
       cos_theta_ijk = dot(Rj[j], Rj[k]) / (dj[j]*dj[k])
       scaled_theta = (2.0 .* cos_theta_ijk .- chsfPI) ./ chsfPI
       Aijk = ch_T(scaled_theta, l-1)
       D .+= Aijk .* fc_ij[j] .* fc_ij[k]
    end 
    return D
end

function c_RADFn(Rj, nmax, lmax, Rc)
    dj = norm.(Rj)
    fc_ij = cutoff_func(dj, Rc)
    D = zeros(nmax * lmax)
    for j = 1:length(Rj)-1, k = j+1:length(Rj)
       scaled_rj = (2.0 * dj[j] - Rc) / Rc
       scaled_rk = (2.0 * dj[k] - Rc) / Rc
       Pj = ch_T(scaled_rj, nmax-1) * fc_ij[j]
       Pk = ch_T(scaled_rk, nmax-1) * fc_ij[k]
       cos_theta_ijk = dot(Rj[j], Rj[k]) / (dj[j]*dj[k])
       scaled_theta = (2.0 * cos_theta_ijk - chsfPI) / chsfPI
       Aijk = ch_T(scaled_theta, lmax-1)
       for n = 1:nmax, l = 1:lmax
           D[n + (l-1) * nmax] += Aijk[l] * Pj[n] * Pk[n]
       end
    end 
    return D
end

function c_RADFnnl(Rj, nmax, lmax, Rc)
    dj = norm.(Rj)
    fc_ij = cutoff_func(dj, Rc)
    D = zeros(nmax, nmax, lmax)
    for j = 1:length(Rj)-1, k = j+1:length(Rj)
       scaled_rj = (2.0 * dj[j] - Rc) / Rc
       scaled_rk = (2.0 * dj[k] - Rc) / Rc
       Pj = ch_T(scaled_rj, nmax-1) * fc_ij[j]
       Pk = ch_T(scaled_rk, nmax-1) * fc_ij[k]
       cos_theta_ijk = dot(Rj[j], Rj[k]) / (dj[j]*dj[k])
       scaled_theta = (2.0 * cos_theta_ijk - chsfPI) / chsfPI
       Aijk = ch_T(scaled_theta, lmax-1)
       for n = 1:nmax, np = 1:nmax, l = 1:lmax
           D[n, np, l] += Aijk[l] * Pj[n] * Pk[np]
       end
    end 
    return D[:]
end

function chsf_desc_RADF(Rs, Rc; n=nothing, l=nothing, np=false)
    if n == nothing
        n = 0
    end
    if l == nothing
        l = 0
    end
    if np
        return c_RADFnnl(Rs, n+1, l+1, Rc)
    else
        return c_RADFn(Rs, n+1, l+1, Rc)
    end
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

function chsf_RADF(at, Rc; n=nothing, l=nothing, np=true)
    representation = []
    ni = []
    nR = []
    for (i, j, R) in pairs(neighbourlist(at, Rc))
        push!(ni, i)
        push!(nR, R)
    end
    for i=1:length(at)
        Rs = nR[(ni .== i)]
        push!(representation,chsf_desc_RADF(Rs, Rc, n=n, l=l, np=np))
    end
    return representation
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
