# EMpht.jl
A Julia port of the EMpht.c program, used for fitting Phase-Type distributions via an EM algorithm.

The original C — which is available on [Søren Asmussen's website](https://web.archive.org/web/20180617130551/http://home.math.au.dk/asmus/pspapers.html) — is well documented and has a decent performance for phase-type distributions with a small or moderate number of phases. However it is quite slow for when the number of phases is large (>= 20), and the UX is very old-school Unix. 

This port is much simpler and faster. See the examples given below and the test cases. 

## Examples

To fit a phase-type distribution to some data:

```julia
using Distributions
using EMpht

data = rand(Exponential(1/10), 1_000)  # Generate some data to fit 
sample = EMpht.Sample(obs=data)        # Create an EMpht Sample object with this data
ph = empht(sample, p=5)                # Fit the data using p=5 phases

xGrid = range(0, 8, length=1_00)       # Create a grid to evaluate the density function over
fitPDFs = pdf.(ph, xGrid)              # The probability density function of the fitted phase-type
```

The default structure of the phase-type is "Coxian" (see below for details). 
For large values of ``p`` the "CanonicalForm1" is recommended. 
To impose no structure on the phase-type, use "General", though the results degrade quickly with ``p > 5`` or so.
Another available option is "GeneralisedCoxian".

```julia
phGen = empht(sample, p=20, ph_structure="General")
phCox = empht(sample, p=20, ph_structure="Coxian")
phCF1 = empht(sample, p=50, ph_structure="CanonicalForm1")
```

If the data is not fully observed, i.e. the data is binned (interval-censored), then the Sample object is updated like so:

```julia
# The intervals
int = [1.5 2.0; 2.0 2.5; 2.5 3.0; 3.0 3.5; 3.5 4.0; 4.0 4.5; 4.5 5.0; 5.0 5.5;
        5.5 6.0; 6.0 6.5; 6.5 7.0; 7.0 7.5]
# The number of observations falling into each interval
intweight = [4.0, 34.0, 107.0, 170.0, 202.0, 222.0, 140.0, 77.0, 24.0, 14.0,
        4.0, 2.0]
 # Create an EMpht Sample object with this data
sInt = EMpht.Sample(int=int, intweight=intweight)

# Fitting the interval-censored data
phCF1 = empht(sInt, p=100, ph_structure="CanonicalForm1")
xGrid = range(0, 8, length=1_000)
fitPDFs = pdf.(phCF1, xGrid)
```

To choose the algorithm used to fit the data (see papers below for details):

```julia
phunif = empht(sample, p=5, method=:unif)  # Fit using the uniformization technique (default)
phode = empht(sample, p=5, method=:ode)    # Fit using the more traditional ODE solving technique
```

EMpht.jl can read all necessary information from a JSON file (the number of phases to fit, the special structure of the phase-type, the sample to fit). For example, if you download the Coxian100.json file inside the test directory, the following will launch a fit based on those parameters:

```julia
ph100 = empht("Coxian100.json")
```

## Resources

The relevant papers for the algorithms are:
- S. Asmussen, O. Nerman & M. Olsson, _Fitting phase-type distribution via the EM algorithm_, Scandinavian Journal of Statistics 23, 419-441 (1996), 
- M. Olsson, _Estimation of phase-type distributions from censored data_, Scandinavian Journal of Statistics 23, 443-460 (1996).
- H. Okamura, T. Dohi, K.S. Trivedi, _A refined EM algorithm for PH distributions_, Performance Evaluation 68, 938-954 (2011)
- H. Okamura, T. Dohi, K.S. Trivedi, _Improvement of expectation-maximization algorithm for phase-type distributions with grouped and truncated data_, Appl. Stochastic Models Bus. Ind. 29, 141-156 (2013) 

Some case studies using this package are:
- S. Asmussen, P.J. Laub, H. Yang, _Phase-type models in life insurance: fitting and valuation of equity-linked benefits_, Risks 7(1), 17 pages (2019)
- A. Vuorinen, _The blockchain propagation process: a machine learning and matrix analytic approach_, University of Melbourne Masters Thesis (2019), see [website](https://bitcoin.aapelivuorinen.com/) or [thesis](https://bitcoin.aapelivuorinen.com/thesis.pdf).
