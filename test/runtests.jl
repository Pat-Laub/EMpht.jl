using Distributions
using EMPht
using Random
using StatsBase # for making a binned version of observed data
using Test

function bin_observations(data, number_bins)
        hist = StatsBase.fit(Histogram, data, nbins=15)

        int = hcat(collect(hist.edges[1][1:end-1]),
                   collect(hist.edges[1][2:end]))
        intweight = convert(Array{Float64}, hist.weights)

        return [hist, int, intweight]
end

# Fit a simple exponential distribution using
# both methods (uniformization and the ODE solver)
Random.seed!(1)
trueRate = 10
trueDist = Exponential(1/trueRate)
data = rand(trueDist, 1_000)
s = EMPht.Sample(obs=data)
ph1unif = empht(s, p=1, method=:unif)
@test -ph1unif.T[1,1] ≈ trueRate atol=1
ph1ode = empht(s, p=1, method=:ode)
@test -ph1ode.T[1,1] ≈ trueRate atol=1

# Fit binned samples from a gamma distribution.
Random.seed!(1)
trueDist = Gamma(20, 2/10)
data = rand(trueDist, 1_000)

xGrid = range(0, maximum(data), length=1_000)
truePDFs = pdf.(trueDist, xGrid)

~, int, intweight = bin_observations(data, 15)
s = EMPht.Sample(int=int, intweight=intweight)

phCF1 = empht(s, p=100, ph_structure="CanonicalForm1")
fitPDFs = pdf.(phCF1, xGrid)

@test maximum(abs.(truePDFs - fitPDFs)) <= 1e-1
