# EMpht.jl
A Julia port of the EMpht.c program, used for fitting Phase-Type distributions via an EM algorithm.

The original C — which is available on [Søren Asmussen's website](https://web.archive.org/web/20180617130551/http://home.math.au.dk/asmus/pspapers.html) — is well documented and has a decent performance for phase-type distributions with a small or moderate number of phases. However it is quite slow for when the number of phases is large (>= 20), and the UX is very old-school Unix. 

This port is much simpler and faster. EMpht.jl reads all necessary information from a JSON file (the number of phases to fit, the special structure of the phase-type, the sample to fit); see the example*.json files which are ports of the examples in the EMpht.c user's guide. It is faster because of it uses: i) a fast matrix multiplication implementation, ii) a lower-order ODE solver, iii) more of a modern CPU's capabilities.

The relevant papers for the algorithms are:
- S. Asmussen, O. Nerman & M. Olsson, _Fitting phase-type distribution via the EM algorithm_, Scandinavian Journal of Statistics 23, 419-441 (1996), 
- M. Olsson, _Estimation of phase-type distributions from censored data_, Scandinavian Journal of Statistics 23, 443-460 (1996).
- H. Okamura, T. Dohi, K.S. Trivedi, _A refined EM algorithm for PH distributions_, Performance Evaluation 68, 938-954 (2011)

Some case studies using this package are:
- S. Asmussen, P.J. Laub, H. Yang, _Phase-type models in life insurance: fitting and valuation of equity-linked benefits_, Risks 7(1), 17 pages (2019)
- A. Vuorinen, _The blockchain propagation process: a machine learning and matrix analytic approach_, University of Melbourne Masters Thesis (2019) [website](https://bitcoin.aapelivuorinen.com/) [thesis](https://bitcoin.aapelivuorinen.com/thesis.pdf).
