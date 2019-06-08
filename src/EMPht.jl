module EMPht

using Dates
using LinearAlgebra
using Random
using Statistics

export empht
# export loglikelihoodcensored
# export parse_settings

include("sample.jl")
include("phasetype.jl")
include("utils.jl")
include("ode.jl")
include("uniformization.jl")

function em_iterate(name, s, fit, ph_structure, method, max_iter, timeout,
        verbose)
    p = fit.p

    # Count the total of all weight.
    sumOfWeights = sum(s.obsweight) + sum(s.censweight) + sum(s.intweight)

    # Sort the observations.
    perm = sortperm(s.obs)
    obsSort = s.obs[perm]
    obsweightSort = s.obsweight[perm]
    s = Sample(obsSort, obsweightSort, s.cens, s.censweight, s.int, s.intweight)

    start = now()
    endTime = start+Dates.Minute(round(timeout))

    save_progress(name, s, fit, start)

    ll = 0

    numPlots = 0
    for iter = 1:max_iter
        ##  The expectation step!
        Bs = zeros(p); Zs = zeros(p); Ns = zeros(p, p+1)

        if length(s.obs) > 0
            if method == :unif
                e_step_observed_uniform!(s, fit, Bs, Zs, Ns)
            elseif method == :ode
                e_step_observed_ode!(s, fit, Bs, Zs, Ns)
            else
                error("Method should be :unif or :ode")
            end
        end

        if length(s.cens) > 0 || length(s.int) > 0
            if method == :unif
                e_step_censored_uniform!(s, fit, Bs, Zs, Ns)
            elseif method == :ode
                e_step_censored_ode!(s, fit, Bs, Zs, Ns)
            else
                error("Method should be :unif or :ode")
            end
        end

        ## The maximisation step!
        π_next = max.(Bs ./ sumOfWeights, 0)
        t_next = max.(Ns[:,end] ./ Zs, 0)
        t_next[isnan.(t_next)] .= 0

        T_next = zeros(p,p)
        for i=1:p
            T_next[i,:] = max.(Ns[i,1:end-1] ./ Zs[i], 0)
            T_next[i,isnan.(T_next[i,:])] .= 0
            T_next[i,i] = -(t_next[i] + sum(T_next[i,:]))
        end

        # Remove any numerical instabilities.
        π_next = max.(π_next, 0)
        π_next /= sum(π_next)

        fit = PhaseType(π_next, T_next, t_next)

        # The CanonicalForm1 phase-type is a bit special. It needs to have an
        # ordering to its diagonal of T, and the EM steps will probably break
        # this, so it needs to be sorted on every iteration.
        if ph_structure == "CanonicalForm1"
            fit = sort_into_canonical_form(fit)
        end

        remaining = endTime - now()
        if remaining.value < 0 || iter >= max_iter
            ll = save_progress(name, s, fit, start)
            seconds = round(now() - start, Dates.Second)
            duration = Dates.canonicalize(Dates.CompoundPeriod(seconds))
            if verbose
                println("Quitting after $duration/$iter iterations with ll = ",
                    round(Float64(ll), sigdigits=5))
            end
            break
        else
            if verbose
                ll = loglikelihoodcensored(s, fit)
                seconds = round(remaining, Dates.Second)
                duration = Dates.canonicalize(Dates.CompoundPeriod(seconds))
                println("Iteration: $iter/$max_iter\t",
                    "Timeout in $duration\t",
                    "Loglikelihood: ", round(Float64(ll), sigdigits=5))
            end
        end
    end

    fit
end

function empht(s; p=1, ph_structure="Coxian", name="", continueFit=false,
        method=:unif, max_iter=100, timeout=1, verbose=false)

    if verbose
        println("(p,ph_structure,name,method,continueFit,max_iter,timeout) = ",
            ((p, ph_structure, name, method, continueFit, max_iter, timeout)))
    end

    # Check we don't just have right-censored obs, since this blows things up.
    if length(s.obs) == 0 && length(s.cens) > 0 && length(s.int) == 0
        error("Can't just have right-censored observations!")
    end

    # If not continuing previous fit, remove any left-over output files.
    if ~continueFit
        rm(string(name, "_loglikelihood.csv"), force=true)
        rm(string(name, "_fit.csv"), force=true)
    end

    fit = initial_phasetype(name, p, ph_structure, continueFit, s, verbose)

    if verbose && p <= 10
        println("first pi is $(fit.π), first T is $(fit.T)\n")
    end

    em_iterate(name, s, fit, ph_structure, method, max_iter, timeout, verbose)
end

function empht(settings_filename::String; method=:unif, verbose=false)
    # Read in details for the fit from the settings file.
    s, p, ph_structure, name, continueFit, max_iter, timeout =
        parse_settings(settings_filename, verbose)
    empht(s, p=p, ph_structure=ph_structure, name=name, continueFit=continueFit,
            method=method, max_iter=max_iter, timeout=timeout, verbose=verbose)
end

end # module
