# Calculate the integral of s=0 to t of exp(T(t-s)) beta alpha exp(T s) ds
# using the uniformized version of T (specified by P and r). See equation (44)
# of Okamura et al (2012), Improvement of expectation-maximization algorithm
# for phase-type distributions with grouped and truncated data.
function conv_int_unif(r, P, t, beta, alpha)
    p = size(P, 1)
    ϵ = 1e-3
    R = quantile(Poisson(r * t), 1-ϵ)

    betas = Array{Float64}(undef, p, R+1)
    betas[:,1] = beta

    for u = 1:R
        betas[:,u+1] = P * betas[:,u]
    end

    poissPDFs = pdf.(Poisson(r*t), 1:R+1)

    alphas = Array{Float64}(undef, p, R+1)
    alphas[:, R+1] = poissPDFs[R+1] .* alpha'

    for u = (R-1):-1:0
        alphas[:, u+1] = alphas[:, u+2]' * P + poissPDFs[u+1] .* alpha'
    end

    Υ = zeros(p, p)
    for u = 0:R
        Υ += betas[:, u+1] * alphas[:, u+1]' ./ r
    end

    Υ
end

function e_step_observed_uniform!(s::Sample, fit::PhaseType,
        Bs::AbstractArray{Float64}, Zs::AbstractArray{Float64},
        Ns::AbstractArray{Float64})
    p = fit.p

    K = length(s.obs)
    deltaTs = diff([0; s.obs])

    fs = zeros(p, K+1)
    bs = zeros(p, K+1)
    fs[:,1] = fit.π
    bs[:,1] = fit.t

    for k = 1:K
        expTDeltaT = exp(fit.T * deltaTs[k])
        fs[:,k+1] = fs[:,k]' * expTDeltaT
        bs[:,k+1] = expTDeltaT * bs[:,k]
    end

    cs = zeros(p, K)
    cs[:,K] = fit.π / dot(fit.π, bs[:,K+1])
    for k = (K-1):-1:1
        cs[:,k] = cs[:,k+1]' * exp(fit.T * deltaTs[k+1]) +
            fit.π' / dot(fit.π, bs[:,k+1])
    end

    H = zeros(p, p)
    r = 1.01 * maximum(abs.(diag(fit.T)))
    P = I + (fit.T ./ r)

    for k = 1:K
        H += s.obsweight[k] .* conv_int_unif(r, P, deltaTs[k], bs[:,k], cs[:,k])
    end

    for k = 1:K
        Bs[:] = Bs[:] + s.obsweight[k] .* (fit.π .* bs[:,k+1]) ./ dot(fit.π, bs[:,k+1])
        Ns[:,end] = Ns[:,end] + s.obsweight[k] .* (fs[:,k+1] .* fit.t) ./ dot(fit.π, bs[:,k+1])
    end

    Zs[:] = Zs[:] + diag(H)
    Ns[:,1:p] = Ns[:,1:p] + fit.T .* (H') .* (1 .- Matrix{Float64}(I, p, p))
end

function e_step_censored_uniform!(s::Sample, fit::PhaseType,
        Bs::AbstractArray{Float64}, Zs::AbstractArray{Float64},
        Ns::AbstractArray{Float64})
    p = fit.p

    int = s.int
    intweight = s.intweight

    # For some reason, we need an interval which starts at 0, so we just
    # add one in with zero weight attached to it.
    if length(s.int) > 0 && minimum(s.int) > 0
        int = vcat([0, minimum(int)]', int)
        intweight = vcat(0, intweight)
    end

    K = size(int, 1)
    deltaTs = int[:,2] - int[:,1]

    barfs = zeros(p, K+1)
    tildefs = zeros(p, K)
    barbs = zeros(p, K+1)
    tildebs = zeros(p, K)
    ws = zeros(K+1)
    N = sum(intweight)
    U = 0

    barfs[:,1] = fit.π' * inv(-fit.T)
    barbs[:,1] = ones(p)

    for k = 1:K
        expTDeltaT = exp(fit.T * deltaTs[k])

        barfs[:,k+1] = barfs[:,k]' * expTDeltaT
        tildefs[:,k] = barfs[:,k] - barfs[:,k+1]
        barbs[:,k+1] = expTDeltaT * barbs[:,k]
        tildebs[:,k] = barbs[:,k] - barbs[:,k+1]

        U += fit.π' * tildebs[:,k]

        ws[k] = intweight[k] / (fit.π' * tildebs[:,k])
    end
    U += fit.π' * barbs[:,K+1]
    ws[K+1] = 0

    cs = zeros(p, K)
    cs[:,K] = (ws[K+1] - ws[K]) .* fit.π'
    for k = (K-1):-1:1
        cs[:,k] = cs[:,k+1]' * exp(fit.T * deltaTs[k+1]) +
            (ws[k+1] - ws[k]) .* fit.π'
    end

    H = zeros(p, p)
    r = 1.01 * maximum(abs.(diag(fit.T)))
    P = I + (fit.T ./ r)

    for k = 1:K
        H += ws[k] .* ones(p) * tildefs[:,k]' +
            conv_int_unif(r, P, deltaTs[k], barbs[:,k], cs[:,k])
    end
    H += ws[K+1] .* ones(p) * barfs[:,K+1]'

    # Step 4
    for k = 1:K
        Bs[:] = Bs[:] + intweight[k] .* (fit.π .* tildebs[:,k]) ./
            (fit.π' * tildebs[:,k])
        Ns[:,end] = Ns[:,end] + intweight[k] .* (tildefs[:,k] .* fit.t) ./
            (tildefs[:,k]' * fit.t)
    end

    Zs[:] = Zs[:] + diag(H)
    Ns[:,1:p] = Ns[:,1:p] + fit.T .* (H') .* (1 .- Matrix{Float64}(I, p, p))
end
