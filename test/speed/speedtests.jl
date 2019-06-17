using BenchmarkTools
using Distributions
using EMPht
using Random

# Fit a simple exponential distribution using
# both methods (uniformization and the ODE solver)
trueRate = 10
trueDist = Exponential(1/trueRate)

Random.seed!(1)
sObs = EMPht.Sample(obs=rand(trueDist, 1_000))

println("Fitting Exponential data using uniformization")
empht(sObs, p=1, method=:unif, max_iter=1, timeout=1e10)
@btime empht(sObs, p=1, method=:unif, max_iter=1e2, timeout=1e10)

println("Fitting Exponential data using ODE")
empht(sObs, p=1, method=:ode, max_iter=1, timeout=1e10)
@btime empht(sObs, p=1, method=:ode, max_iter=1e2, timeout=1e10)

# Fit binned samples from a gamma distribution.
trueDist = Gamma(20, 2/10)

int = [1.5 2.0; 2.0 2.5; 2.5 3.0; 3.0 3.5; 3.5 4.0; 4.0 4.5; 4.5 5.0; 5.0 5.5;
        5.5 6.0; 6.0 6.5; 6.5 7.0; 7.0 7.5]
intweight = [4.0, 34.0, 107.0, 170.0, 202.0, 222.0, 140.0, 77.0, 24.0, 14.0,
        4.0, 2.0]
sInt = EMPht.Sample(int=int, intweight=intweight)

println("Fitting binned Gamma distribution with CF1(100) using uniformization")
empht(sInt, p=100, ph_structure="CanonicalForm1",
        max_iter=1, timeout=1e10, method=:unif)
@btime empht(sInt, p=100, ph_structure="CanonicalForm1",
        max_iter=50, timeout=1e10, method=:unif)

println("Fitting binned Gamma distribution with CF1(20) using ODE")
empht(sInt, p=20, ph_structure="CanonicalForm1", max_iter=1,
                timeout=1e10, method=:ode)
@btime empht(sInt, p=20, ph_structure="CanonicalForm1", max_iter=50,
                timeout=1e10, method=:ode)

## Test that some of the internal functions work as expected.
πVec = [0.005632862236468221, 0.00898302181442919, 0.007418596691981033,
        0.020026201266915976, 0.04062880012941406, 0.05812175206666447,
        0.04260242040415281, 0.1654039105298598, 0.3888838419368729,
        0.2622985929232415]
T = [-1.7486539983379394 1.7486539983379394 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0;
        0.0 -1.9875507782505892 1.9875507782505892 0.0 0.0 0.0 0.0 0.0 0.0 0.0;
        0.0 0.0 -1.9962263354076826 1.9962263354076826 0.0 0.0 0.0 0.0 0.0 0.0;
        0.0 0.0 0.0 -2.064044688471394 2.064044688471394 0.0 0.0 0.0 0.0 0.0;
        0.0 0.0 0.0 0.0 -2.1489226352793827 2.1489226352793827 0.0 0.0 0.0 0.0;
        0.0 0.0 0.0 0.0 0.0 -2.2809066095317565 2.2809066095317565 0.0 0.0 0.0;
        0.0 0.0 0.0 0.0 0.0 0.0 -2.5379722045000195 2.5379722045000195 0.0 0.0;
        0.0 0.0 0.0 0.0 0.0 0.0 0.0 -2.763746562995887 2.763746562995887 0.0;
        0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 -3.082614321671955 3.082614321671955;
        0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 -3.910958867504505]
t = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 3.910958867504505]
ph = EMPht.PhaseType(πVec, T, t)

u = zeros(ph.p * ph.p)
du = similar(u)

println("Running ode_observations")
EMPht.ode_observations!(du, u, ph, 10.0)
@btime EMPht.ode_observations!(du, u, ph, 10.0)

println("Running ode_censored")
EMPht.ode_censored!(du, u, ph, 10.0)
@btime EMPht.ode_censored!(du, u, ph, 10.0)

println("Running c_integrand")
EMPht.c_integrand(5.0, ph, 10.0)
@btime EMPht.c_integrand(5.0, ph, 10.0)

println("Running d_integrand")
EMPht.d_integrand(5.0, ph, 10.0)
@btime EMPht.d_integrand(5.0, ph, 10.0)

p = ph.p; Bs = zeros(p); Zs = zeros(p); Ns = zeros(p, p+1)

println("Running e_step_observed_ode")
EMPht.e_step_observed_ode!(sObs, ph, Bs, Zs, Ns)
@btime EMPht.e_step_observed_ode!(sObs, ph, Bs, Zs, Ns)

p = ph.p; Bs = zeros(p); Zs = zeros(p); Ns = zeros(p, p+1)

println("Running e_step_censored_ode")
EMPht.e_step_censored_ode!(sInt, ph, Bs, Zs, Ns)
@btime EMPht.e_step_censored_ode!(sInt, ph, Bs, Zs, Ns)
