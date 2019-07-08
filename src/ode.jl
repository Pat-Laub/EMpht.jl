using OrdinaryDiffEq
using HCubature

function ode_observations!(du::AbstractArray{Float64},
        u::AbstractArray{Float64}, fit::PhaseType, t::Float64)
    # dc = T * C + t * a
    a = fit.π' * exp(fit.T * t)
    du[:] = vec(fit.T * reshape(u, fit.p, fit.p) + fit.t * a)
end

function ode_censored!(du::AbstractArray{Float64}, u::AbstractArray{Float64},
        fit::PhaseType, t::Float64)
    # dc = T * C + 1 * a
    a = fit.π' * exp(fit.T * t)
    du[:] = vec(fit.T * reshape(u, fit.p, fit.p) + ones(fit.p) * a)
end

function c_integrand(u, fit, y)
    # Compute the two vector terms in the integrand
    first = fit.π' * exp(fit.T * u)
    second = exp(fit.T * (y-u)) * fit.t

    # Construct the matrix of integrands using outer product
    # then reshape it to a vector.
    C = second * first
    vec(C)
end

function d_integrand(u, fit, y)
    # Compute the two vector terms in the integrand
    first = fit.π' * exp(fit.T * u)
    second = exp(fit.T * (y-u)) * ones(fit.p)

    # Construct the matrix of integrands using outer product
    # then reshape it to a vector.
    D = second * first
    vec(D)
end

function e_step_observed_ode!(s::Sample, fit::PhaseType,
        Bs::AbstractArray{Float64}, Zs::AbstractArray{Float64},
        Ns::AbstractArray{Float64})
    # Setup initial conditions.
    p = fit.p
    u0 = zeros(p*p)

    # Run the ODE solver.
    prob = ODEProblem(ode_observations!, u0, (0.0, maximum(s.obs)), fit)
    sol = solve(prob, Tsit5())
    Id = Matrix{Float64}(I, p, p)

    for k = 1:length(s.obs)
        weight = s.obsweight[k]

        expTy = exp(fit.T * s.obs[k])
        a = transpose(fit.π' * expTy)
        b = expTy * fit.t

        u = sol(s.obs[k])
        C = reshape(u, p, p)

        if minimum(C) < 0
            (C,err) = hquadrature(x -> c_integrand(x, fit, s.obs[k]), 0,
            s.obs[k], rtol=1e-1, maxevals=500)
            C = reshape(C, p, p)
        end

        denom = fit.π' * b
        Bs[:] = Bs[:] + weight * (fit.π .* b) / denom
        Zs[:] = Zs[:] + weight * diag(C) / denom
        Ns[:,1:p] = Ns[:,1:p] + weight * (fit.T .* C' .* (1 .- Id)) / denom
        Ns[:,p+1] = Ns[:,end] + weight * (fit.t .* a) / denom
    end
end

function e_step_censored_ode!(s::Sample, fit::PhaseType,
        Bs::AbstractArray{Float64}, Zs::AbstractArray{Float64},
        Ns::AbstractArray{Float64})
    # Setup initial conditions.
    p = fit.p
    u0 = zeros(p*p)

    # Should tell the ODE solver to evaluate all right and interval censored
    # points.
    allcens = vcat(s.cens, vec(s.int))

    # Run the ODE solver.
    prob = ODEProblem(ode_censored!, u0, (0.0, maximum(allcens)), fit)
    sol = solve(prob, Tsit5())

    for k = 1:length(s.cens)
        weight = s.censweight[k]

        h = exp(fit.T * s.cens[k]) * ones(p)

        u = sol(s.cens[k])
        D = reshape(u, p, p)

        if minimum(D) < 0
            (D, err) = hquadrature(x -> d_integrand(x, fit, s.cens[k]), 0,
                s.cens[k], rtol=1e-1, maxevals=500)
            D = reshape(D, p, p)
        end

        denom = fit.π' * h
        Bs[:] = Bs[:] + weight * (fit.π .* h) / denom
        Zs[:] = Zs[:] + weight * diag(D) / denom
        Ns[:,1:p] = Ns[:,1:p] + weight * (fit.T .* transpose(D) .*
            (1 .- Matrix{Float64}(I, p, p))) / denom
    end

    a = 0; h = 0; g = 0; D = 0; expTRight = 0

    for k = 1:size(s.int, 1)
        weight = s.intweight[k]
        left, right = s.int[k,:]

        if k > 1 && left == s.int[k-1,2]
            a_left = a
            h_left = h
            g_left = g
            D_left = D
            expTLeft = expTRight
        else
            expTLeft = exp(fit.T * left)
            a_left = transpose(fit.π' * expTLeft)
            h_left = expTLeft * ones(p)
            g_left = try
                        transpose(fit.T) \ (a_left - fit.π)
                    catch
                        transpose(transpose(a_left - fit.π) * pinv(fit.T))
                    end

            u_left = sol(left)
            D_left = reshape(u_left, p, p)
            if minimum(D_left) < 0 && iscoxian(fit)
                minD = minimum(D_left)
                (D_left, err) = hquadrature(x -> d_integrand(x, fit, left), 0,
                    left, rtol=1e-1, maxevals=500)
                D_left = reshape(D_left, p, p)
            end
        end

        expTRight = exp(fit.T * right)
        a = transpose(fit.π' * expTRight)
        h = expTRight * ones(p)
        g = try
                transpose(fit.T) \ (a - fit.π)
            catch
                transpose(transpose(a - fit.π) * pinv(fit.T))
            end

        u_right = sol(right)
        D = reshape(u_right, p, p)
        if minimum(D) < 0 && iscoxian(fit)
            (D, err) = hquadrature(x -> d_integrand(x, fit, right), 0, right,
                rtol=1e-1, maxevals=500)
            D = reshape(D, p, p)
        end

        T_legal = fit.T .> 0

        denom = fit.π' * (expTLeft - expTRight) * ones(fit.p)

        Bs[:] = Bs[:] + weight * (fit.π .* (h_left - h)) / denom
        Zs[:] = Zs[:] + weight * (g - g_left + diag(D_left) - diag(D)) / denom
        Ns[:,1:p] = Ns[:,1:p] + weight * T_legal .* (fit.T .* ((g - g_left) .+
            transpose(D_left - D))) / denom
        Ns[:,p+1] = Ns[:,p+1] + weight * (fit.t .* (g - g_left)) / denom
    end
end
