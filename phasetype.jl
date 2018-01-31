## Definition of a phase-type distribution, and related functions.
using Distributions
import Distributions: cdf, insupport, logpdf, pdf
import Base: sum, mean, median, maximum, minimum, quantile, std, var, cov, cor

macro check_args(D, cond)
    quote
        if !($(esc(cond)))
            throw(ArgumentError(string(
                $(string(D)), ": the condition ", $(string(cond)), " is not satisfied.")))
        end
    end
end

struct PhaseType <: ContinuousUnivariateDistribution
    # Defining properties
    π # entry probabilities
    T # transition probabilities

    # Derived properties
    t # exit probabilities
    p # number of phases
    Tinv # inverse of T matrix

    function PhaseType(π, T, t, p, Tinv)
        @check_args(PhaseType, all(π .>= zero(π[0])))
        @check_args(PhaseType, isapprox(sum(π), 1.0, atol=1e-4))
        @check_args(PhaseType, p == length(π) && p == length(t) && all(p .== size(T)))
        @check_args(PhaseType, all(t .>= zero(t[0])))
        @check_args(PhaseType, all(isapprox(t, -T*ones(p))))
        new(π, T, t, p, Tinv)
    end
end

PhaseType(π, T) = PhaseType(π, T, -T*ones(length(π)), length(π), inv(T))

minimum(d::PhaseType) = 0
maximum(d::PhaseType) = Inf
insupport(d::PhaseType, x::Real) = x > 0 && x < Inf

pdf(d::PhaseType, x::Real) = transpose(d.π) * expm(d.T * x) * d.t
logpdf(d::PhaseType, x::Real) = log(pdf(d, x))
cdf(d::PhaseType, x::Real) = 1 - transpose(d.π) * expm(d.T * x) * ones(d.p)

mean(d::PhaseType) = -transpose(d.π) * d.Tinv * ones(d.p)
