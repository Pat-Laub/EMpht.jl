using Distributions
using Plots
using StatsBase # for making a binned version of observed data

include("empht.jl")

function bin_observations(data, number_bins)
        hist = StatsBase.fit(Histogram, data, nbins=15)
        plot(normalize(hist), label="Interval Data")

        int = hcat(collect(hist.edges[1][1:end-1]), collect(hist.edges[1][2:end]))
        intweight = convert(Array{Float64}, hist.weights)

        return [hist, int, intweight]
end

### Fit a simple exponential distribution
Random.seed!(1)
trueDist = Exponential(10)
data = rand(trueDist, 1_000)

histogram(data, normalize=:pdf, bins=40, label="Data")
xGrid = range(0, maximum(data), length=1_000)
plot!(xGrid, pdf.(trueDist, xGrid), label="True PDF", linewidth=3)

s = Sample(obs=data)

ph1unif = em(s, p=5, ph_structure="General", max_iter=100, timeout=1,
        verbose=true, method=:unif)
ph1ode = em(s, p=5, ph_structure="General", max_iter=100, timeout=1,
        verbose=true, method=:ode)
plot!(xGrid, pdf.(ph1unif, xGrid),
        label="PH(General,$(ph1unif.p))[Uniformization]", linewidth=3)
plot!(xGrid, pdf.(ph1ode, xGrid),
        label="PH(General,$(ph1ode.p))[ODE]", linewidth=3)

ph2 = em(s, p=5, ph_structure="CanonicalForm1", max_iter=100,
        timeout=1, verbose=true)
plot!(xGrid, pdf.(ph2, xGrid), label="PH(CF1,$(ph2.p))", linewidth=3)


### Fit a binned version of the data
hist = StatsBase.fit(Histogram, data, nbins=15)
plot(normalize(hist), label="Interval Data")

edges = hist.edges[1]
weights = hist.weights
ints = hcat(collect(edges[1:end-1]), collect(edges[2:end]))
weights = convert(Array{Float64}, weights)
s = Sample(int=ints, intweight=weights)

ph1 = em(s, p=5, ph_structure="General", max_iter=100, timeout=1,
        verbose=true)
plot!(xGrid, pdf.(ph1, xGrid),
        label="PH(General,$(ph1unif.p))[Uniformization]", linewidth=3)
ph2 = em(s, p=5, ph_structure="CanonicalForm1", max_iter=100,
        timeout=1, verbose=true)
plot!(xGrid, pdf.(ph2, xGrid), label="PH(CF1,$(ph2.p))", linewidth=3)


### Fit a gamma distribution.
Random.seed!(1)
trueDist = Gamma(20, 2/10)
data = rand(trueDist, 1_000)

histogram(data, normalize=:pdf, bins=40, label="Data")
xGrid = range(0, maximum(data), length=1_000)
plot!(xGrid, pdf.(trueDist, xGrid), label="True PDF", linewidth=3)

s = Sample(obs=data)

ph1 = em(s, p=20, ph_structure="General", max_iter=200, timeout=2, verbose=true)
plot!(xGrid, pdf.(ph1, xGrid), label="PH(Gen,$(ph1.p))", linewidth=3)

ph2 = em(s, p=10, ph_structure="CanonicalForm1", max_iter=200, timeout=1, verbose=true)
plot!(xGrid, pdf.(ph2, xGrid), label="PH(CF1,$(ph2.p))", linewidth=3)

bestFit = fit(Gamma, gamms)
plot!(xGrid, pdf.(bestFit, xGrid), label="Best Fit", linewidth=3)


### Fit a binned version of the data
histogram(gamms, normalize=:pdf, bins=40, label="Data")
plot!(xGrid, pdf.(trueDist, xGrid), label="True PDF", linewidth=3)

hist = StatsBase.fit(Histogram, gamms, nbins=15)
plot!(normalize(hist), label="Interval Data", opacity=0.5)

edges = hist.edges[1]
weights = hist.weights
ints = hcat(collect(edges[1:end-1]), collect(edges[2:end]))
weights = convert(Array{Float64}, weights)
s = Sample(int=ints, intweight=weights)

xGrid = range(0, maximum(edges), length=1_000)
ph1ode = em(s, p=12, ph_structure="General", max_iter=100, timeout=1, verbose=true, method=:ode)
ph1unif = em(s, p=12, ph_structure="General", max_iter=100, timeout=1, verbose=true, method=:unif)
ph1 = em(s, p=12, ph_structure="General", max_iter=2_000, timeout=1, verbose=true)
# ph1 = em(s, p=20, ph_structure="General", max_iter=100, timeout=1, verbose=true)
plot!(xGrid, pdf.(ph1, xGrid), label="PH(Gen,$(ph1.p))", linewidth=3)

ph2 = em(s, p=100, ph_structure="CanonicalForm1", max_iter=1_00, timeout=1, verbose=true)
# ph2 = em(s, p=2, ph_structure="CanonicalForm1", max_iter=10, timeout=1, verbose=true)
plot!(xGrid, pdf.(ph2, xGrid), label="PH(CF1,$(ph2.p))", linewidth=3)

# Test 3: Fit a mixture distribution
Random.seed!(1)
gamms1 = rand(Distributions.Gamma(20, 2/10), 1_000)
gamms2 = rand(Distributions.Gamma(100, 1/100), 1_000)
mixture = [gamms1; gamms2]
s = Sample(;obs=mixture)
histogram(mixture, normalize=:pdf, bins=50, label="Data")
xGrid = range(0, maximum(mixture), length=1_000)

ph1 = em(s, p=10, ph_structure="General", timeout=5, verbose=true)
plot!(xGrid, pdf.(ph1, xGrid), label="PH(Gen,$(ph1.p))")

ph2 = em(s, p=40, ph_structure="CanonicalForm1", timeout=5, verbose=true)
plot!(xGrid, pdf.(ph2, xGrid), label="PH(CF1,$(ph2.p))")

ph3 = em(s, p=80, ph_structure="CanonicalForm1", timeout=5, verbose=true)
plot!(xGrid, pdf.(ph3, xGrid), label="PH(CF1,$(ph3.p))")
