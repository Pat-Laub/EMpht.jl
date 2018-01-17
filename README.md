# EMpht.jl
A Julia port of the EMpht.c program, used for fitting Phase-Type distributions via an EM algorithm.

The original C — which is available on [Søren Asmussen's website](http://home.math.au.dk/asmus/pspapers.html) — is well documented and has a decent performance for phase-type distributions with a small or moderate number of phases. However it is quite slow for when the number of phases is large (>= 20), and the UX is very old-school Unix. 

This port is much simpler (~300 lines of Julia compared with ~1700 lines of C), and faster. EMpht.jl reads all necessary information from a JSON file (the number of phases to fit, the special structure of the phase-type, the sample to fit); see the example*.json files which are ports of the examples in the EMpht.c user's guide. It is faster because of it uses: i) a fast matrix multiplication implementation, ii) a lower-order ODE solver, iii) more of a modern CPU's capabilities.

The relevant papers for the algorithm are  S. Asmussen, O. Nerman & M. Olsson, _Fitting phase-type distribution via the EM algorithm_, Scand. J. Statist. 23, 419-441 (1996), and M. Olsson, _Estimation of Phase-Type Distributions from Censored Data_, Scand. J. Statist. 23, 443-460 (1996).
