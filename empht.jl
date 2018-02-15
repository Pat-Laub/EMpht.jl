using Cubature
using JSON
using OrdinaryDiffEq
using Plots; gr();
include("phasetype.jl");

# Definition of a sample which we fit the phase-type distribution to.
struct Sample
    obs::Vector{Float64}
    obsweight::Vector{Float64}
    cens::Vector{Float64}
    censweight::Vector{Float64}
    int::Matrix{Float64}
    intweight::Vector{Float64}

    function Sample(obs::Vector{Float64}, obsweight::Vector{Float64},
            cens::Vector{Float64}, censweight::Vector{Float64},
            int::Matrix{Float64}, intweight::Vector{Float64})
        cond = all(obs .>= 0) && all(obsweight .> 0) && all(cens .>= 0) &&
                all(censweight .> 0) && all(int .>= 0) && all(intweight .> 0)
        if ~cond
            error("Require non-negativity of observations and positivity of weight")
        end
        new(obs, obsweight, cens, censweight, int, intweight)
    end
end

function ode_observations!(t::Float64, u::AbstractArray{Float64}, fit::PhaseType, du::AbstractArray{Float64})
    # dc = T * C + t * a
    a = fit.π' * expm(fit.T * t)
    du[:] = vec(fit.T * reshape(u, fit.p, fit.p) + fit.t * a)
end

function ode_censored!(t::Float64, u::AbstractArray{Float64}, fit::PhaseType, du::AbstractArray{Float64})
    # dc = T * C + 1 * a
    a = fit.π' * expm(fit.T * t)
    du[:] = vec(fit.T * reshape(u, fit.p, fit.p) + ones(fit.p) * a)
end

function loglikelihoodcensored(s::Sample, fit::PhaseType)
    ll = 0.0

    for k = 1:length(s.obs)
        ll += s.obsweight[k] * log(pdf(fit, s.obs[k]))
    end

    for k = 1:length(s.cens)
        ll += s.censweight[k] * log(1 - cdf(fit, s.cens[k]))
    end

    for k = 1:size(s.int, 1)
        ll_k = log( fit.π' * (expm(fit.T * s.int[k,1]) - expm(fit.T * s.int[k,2]) ) * ones(fit.p) )
        ll += s.intweight[k] * ll_k
    end

    ll
end

function parse_settings(settings_filename::String)
    # Check the file exists.
    if ~isfile(settings_filename)
        error("Settings file $settings_filename not found.")
    end

    # Check the input file is a json file.
    if length(settings_filename) < 6 || settings_filename[end-4:end] != ".json"
        error("Require a settings file as 'filename.json'.")
    end

    # Read in the properties of this fit (e.g. number of phases, PH structure)
    println("Reading settings from $settings_filename")
    settings = JSON.parsefile(settings_filename, use_mmap=false)

    name = get(settings, "Name", basename(settings_filename)[1:end-5])
    p = get(settings, "NumberPhases", 15)
    ph_structure = get(settings, "Structure", p < 20 ? "General" : "Coxian")
    continueFit = get(settings, "ContinuePreviousFit", true)
    num_iter = get(settings, "NumberIterations", 1_000)
    timeout = get(settings, "TimeOut", 30)

    # Set the seed for the random number generation if requested.
    if haskey(settings, "RandomSeed")
        srand(settings["RandomSeed"])
    else
        srand(1)
    end

    # Fill in the default values for the sample.
    s = settings["Sample"]

    obs = haskey(s, "Uncensored") ? Vector{Float64}(s["Uncensored"]["Observations"]) : Vector{Float64}()
    cens = haskey(s, "RightCensored") ? Vector{Float64}(s["RightCensored"]["Cutoffs"]) : Vector{Float64}()
    int = haskey(s, "IntervalCensored") ? Matrix{Float64}(transpose(hcat(s["IntervalCensored"]["Intervals"]...))) : Matrix{Float64}(0, 0)

    # Set the weight to 1 if not specified.
    obsweight = length(obs) > 0 && haskey(s["Uncensored"], "Weights") ? Vector{Float64}(s["Uncensored"]["Weights"]) : ones(length(obs))
    censweight = length(cens) > 0 && haskey(s["RightCensored"], "Weights") ? Vector{Float64}(s["RightCensored"]["Weights"]) : ones(length(cens))
    intweight = length(int) > 0 && haskey(s["IntervalCensored"], "Weights") ? Vector{Float64}(s["IntervalCensored"]["Weights"]) : ones(length(int))

    s = Sample(obs, obsweight, cens, censweight, int, intweight)

    (name, p, ph_structure, continueFit, num_iter, timeout, s)
end

function initial_phasetype(name::String, p::Int, ph_structure::String, continueFit::Bool, s::Sample)
    # If there is a <Name>_phases.csv then read the data from there.
    phases_filename = string(name, "_fit.csv")

    if continueFit && isfile(phases_filename)
        println("Continuing fit in $phases_filename")
        phases = readdlm(phases_filename)
        π = phases[1:end, 1]
        T = phases[1:end, 2:end]
        if length(π) != p || size(T) != (p, p)
            error("Error reading $phases_filename, expecting $p phases")
        end
        t = -T * ones(p)

    else # Otherwise, make a random start for the matrix.
        println("Using a random starting value")
        if ph_structure == "General"
            π_legal = trues(p)
            T_legal = trues(p, p)
        elseif ph_structure == "Coxian"
            π_legal = 1:p .== 1
            T_legal = diagm(ones(p-1), 1) .> 0
        elseif ph_structure == "GeneralisedCoxian"
            π_legal = trues(p)
            T_legal = diagm(ones(p-1), 1) .> 0
        else
            error("Nothing implemented for phase-type structure $ph_structure")
        end

        # Create a structure using [0.1, 1] uniforms.
        t = (0.9 * rand(p) + 0.1)

        π = (0.9 * rand(p) + 0.1)
        π[.~π_legal] = 0
        π /= sum(π)

        T = (0.9 * rand(p, p) + 0.1)
        T[.~T_legal] = 0
        T -= diagm(T*ones(p) + t)

        # Rescale t and T using the same scaling as in the EMPHT.c program.
        if length(s.obs) > min(length(s.cens), size(s.int, 1))
            scalefactor = median(s.obs)
        elseif size(s.int, 1) > length(s.cens)
            scalefactor = median(s.int[:,2])
        else
            scalefactor = median(s.cens)
        end

        t *= p / scalefactor
        T *= p / scalefactor
    end

    PhaseType(π, T)
end

function save_progress(name::String, s::Sample, fit::PhaseType, plotDens::Bool, plotMax::Float64, start::DateTime)
    ll = loglikelihoodcensored(s, fit)
    println("loglikelihood $ll")

    open(string(name, "_loglikelihood.csv"), "a") do f
        mins = (now() - start).value / 1000 / 60
        write(f, "$ll $(round(mins, 4))\n")
    end

    writedlm(string(name, "_fit.csv"), [fit.π fit.T])

    if plotDens
        xVals = linspace(0, plotMax, 100)
        yVals = pdf.(fit, xVals)
        fig = plot!(xVals, yVals)
        png(string(name, "_fit"))
    end

    ll
end

function c_integrand(u, v, fit, y)
    # Compute the two vector terms in the integrand
    first = fit.π' * expm(fit.T * u)
    second = expm(fit.T * (y-u)) * fit.t

    # Construct the matrix of integrands using outer product
    # then reshape it to a vector.
    C = second * first
    v[:] = vec(C)
end

function d_integrand(u, v, fit, y)
    # Compute the two vector terms in the integrand
    first = fit.π' * expm(fit.T * u)
    second = expm(fit.T * (y-u)) * ones(fit.p)

    # Construct the matrix of integrands using outer product
    # then reshape it to a vector.
    D = second * first
    v[:] = vec(D)
end


function conditional_on_obs!(s::Sample, fit::PhaseType, Bs::AbstractArray{Float64}, Zs::AbstractArray{Float64}, Ns::AbstractArray{Float64})
    # Setup initial conditions.
    p = fit.p
    u0 = zeros(p*p)

    # Run the ODE solver.
    pf = ParameterizedFunction(ode_observations!, fit)
    prob = ODEProblem(pf, u0, (0.0, maximum(s.obs)))
    sol = solve(prob, OwrenZen5())

    for k = 1:length(s.obs)
        weight = s.obsweight[k]

        expTy = expm(fit.T * s.obs[k])
        a = transpose(fit.π' * expTy)
        b = expTy * fit.t

        u = sol(s.obs[k])
        C = reshape(u, p, p)
        if minimum(C) < 0
            (C,err) = hquadrature(p*p, (x,v) -> c_integrand(x, v, fit, s.obs[k]), 0, s.obs[k], reltol=1e-1, maxevals=500)
            C = reshape(C, p, p)
        end

        denom = fit.π' * b
        Bs[:] = Bs[:] + weight * (fit.π .* b) / denom
        Zs[:] = Zs[:] + weight * diag(C) / denom
        Ns[:,1:p] = Ns[:,1:p] + weight * (fit.T .* transpose(C) .* (1-eye(p))) / denom
        Ns[:,p+1] = Ns[:,end] + weight * (fit.t .* a) / denom
    end
end

function conditional_on_cens!(s::Sample, fit::PhaseType, Bs::AbstractArray{Float64}, Zs::AbstractArray{Float64}, Ns::AbstractArray{Float64})
    # Setup initial conditions.
    p = fit.p
    u0 = zeros(p*p)

    # Should tell the ODE solver to evaluate all right and interval censored points.
    allcens = vcat(s.cens, vec(s.int))

    # Run the ODE solver.
    pf = ParameterizedFunction(ode_censored!, fit)
    prob = ODEProblem(pf, u0, (0.0, maximum(allcens)))
    sol = solve(prob, OwrenZen5())

    for k = 1:length(s.cens)
        weight = s.censweight[k]

        h = expm(fit.T * s.cens[k]) * ones(p)

        u = sol(s.cens[k])
        D = reshape(u, p, p)
        if minimum(D) < 0
            (D, err) = hquadrature(p*p, (x,v) -> d_integrand(x, v, fit, s.cens[k]), 0, s.cens[k], reltol=1e-1, maxevals=500)
            D = reshape(D, p, p)
        end

        denom = fit.π' * h
        Bs[:] = Bs[:] + weight * (fit.π .* h) / denom
        Zs[:] = Zs[:] + weight * diag(D) / denom
        Ns[:,1:p] = Ns[:,1:p] + weight * (fit.T .* transpose(D) .* (1-eye(p))) / denom
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
            expTLeft = expm(fit.T * left)
            a_left = transpose(fit.π' * expTLeft)
            h_left = expTLeft * ones(p)
            g_left = transpose(fit.T) \ (a_left - fit.π)

            u_left = sol(left)
            D_left = reshape(u_left, p, p)
            if minimum(D_left) < 0
                (D_left, err) = hquadrature(p*p, (x,v) -> d_integrand(x, v, fit, left), 0, left, reltol=1e-1, maxevals=500)
                D_left = reshape(D_left, p, p)
            end
        end

        expTRight = expm(fit.T * right)
        a = transpose(fit.π' * expTRight)
        h = expTRight * ones(p)
        g = transpose(fit.T) \ (a - fit.π)

        u_right = sol(right)
        D = reshape(u_right, p, p)
        if minimum(D) < 0
            (D, err) = hquadrature(p*p, (x,v) -> d_integrand(x, v, fit, right), 0, right, reltol=1e-1, maxevals=500)
            D = reshape(D, p, p)
        end

        T_legal = fit.T .> 0

        denom = fit.π' * (expTLeft - expTRight) * ones(fit.p)

        if any( weight * (fit.π .* (h_left - h)) / denom .< 0 )
            println("pi = $(fit.π)")
            println("bdiff = $(h_left - h)")
            println("denom = $denom")
        end
        Bs[:] = Bs[:] + weight * (fit.π .* (h_left - h)) / denom
        Zs[:] = Zs[:] + weight * (g - g_left + diag(D_left) - diag(D)) / denom
        Ns[:,1:p] = Ns[:,1:p] + weight * T_legal .* (fit.T .* ((g - g_left) .+ transpose(D_left - D))) / denom
        Ns[:,p+1] = Ns[:,p+1] + weight * (fit.t .* (g - g_left)) / denom
    end
end

function em_iterate(name, s, fit, num_iter, timeout, test_run)
    p = fit.p

    # Count the total of all weight.
    sumOfWeights = sum(s.obsweight) + sum(s.censweight) + sum(s.intweight)

    # Find the largest of the different samples to set appropriate plot size.
    plotMax = 1.1 * mapreduce(l -> length(l) > 0 ? maximum(l) : 0, max, (s.obs, s.cens, s.int))

    start = now()

    save_progress(name, s, fit, true, plotMax, start)

    ll = 0
    fig = plot()
    numPlots = 0
    for iter = 1:num_iter
        if iter == 1 || iter % 5 == 0
            mins = (now() - start).value / 1000 / 60
            println("Starting iteration $iter ($(round(mins, 2)) mins)")
        end

        ##  The expectation step!
        Bs = zeros(p); Zs = zeros(p); Ns = zeros(p, p+1)

        if length(s.obs) > 0
            conditional_on_obs!(s, fit, Bs, Zs, Ns)
        end

        if length(s.cens) > 0 || length(s.int) > 0
            conditional_on_cens!(s, fit, Bs, Zs, Ns)
        end

        if minimum(Bs) < 0
            println("Bs = $Bs")
            error("Bs shouldn't go negative")
        end
        if minimum(Zs) < 0
            error("Zs shouldn't go negative")
        end
        if minimum(Ns) < 0
            println("Ns = $Ns")
            error("Ns shouldn't go negative $(minimum(Ns))")
        end

        ## The maximisation step!
        π_next = Bs ./ sumOfWeights
        t_next = Ns[:,end] ./ Zs

        T_next = zeros(p,p)
        for i=1:p
            T_next[i,:] = Ns[i,1:end-1] ./ Zs[i]
            T_next[i,i] = -(t_next[i] + sum(T_next[i,:]))
        end

        # Remove any numerical instabilities.
        π_next = max.(π_next, 0)
        π_next /= sum(π_next)

        fit = PhaseType(π_next, T_next)

        if (now() - start) > Dates.Minute(round(timeout))
            ll = save_progress(name, s, fit, ~test_run, plotMax, start)
            println("Quitting due to going overtime after $iter iterations.")
            break
        end

        # Plot each iteration at the beginning
        saveplot = ~test_run && (iter % 10 == 0) && numPlots < 20
        numPlots += saveplot
        ll = save_progress(name, s, fit, saveplot, plotMax, start)
    end

    ll
end

function em(settings_filename::String)
    # Read in details for the fit from the settings file.
    name, p, ph_structure, continueFit, num_iter, timeout, s = parse_settings(settings_filename)
    println("name, p, ph_structure, continueFit, num_iter, timeout = $((name, p, ph_structure, continueFit, num_iter, timeout))")
    println("Starting with ODE solver, but switching to quadrature on failure")

    # Check we don't just have right-censored obs, since this blows things up.
    if length(s.obs) == 0 && length(s.cens) > 0 && length(s.int) == 0
        error("Can't just have right-censored observations!")
    end

    # If not continuing previous fit, remove any left-over output files.
    if ~continueFit
        rm(string(name, "_loglikelihood.csv"), force=true)
        rm(string(name, "_fit.csv"), force=true)
    end

    # If we start randomly, give it a go from 3 locations before fully running.
    if ~continueFit
        fit1 = initial_phasetype(name, p, ph_structure, continueFit, s)
        fit2 = initial_phasetype(name, p, ph_structure, continueFit, s)
        fit3 = initial_phasetype(name, p, ph_structure, continueFit, s)

        ll1 = em_iterate(name, s, fit1, 30, timeout/4, true)
        ll2 = em_iterate(name, s, fit2, 30, timeout/4, true)
        ll3 = em_iterate(name, s, fit3, 30, timeout/4, true)

        maxll = maximum([ll1, ll2, ll3])
        println("Best ll was $maxll out of $([ll1, ll2, ll3])")
        if ll1 == maxll
            println("Using first guess")
            fit = fit1
        elseif ll2 == maxll
            println("Using second guess")
            fit = fit2
        else
            println("Using third guess")
            fit = fit3
        end
    else
        fit = initial_phasetype(name, p, ph_structure, continueFit, s)
    end

    if p <= 10
        println("first pi is $(fit.π), first T is $(fit.T)\n")
    end

    em_iterate(name, s, fit, num_iter, timeout, false)
end
