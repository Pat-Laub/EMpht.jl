using JSON
using OrdinaryDiffEq
using Plots
gr()

struct PhaseType
    π
    T
    t
end

struct Sample
    obs
    obsWeights
    cens
    censWeights
    ints
    intsWeights
end

function ode_observations(_, u, fit, du)
    p = length(fit.π)

    # da = a' * T (where the first p components of u are 'a')
    du[1:p] = u[1:p]' * fit.T

    # db = T * b (where the next p components of u are 'b')
    du[p+1:2p] = fit.T * u[p+1:2p]

    ## dc = T * [c_1 c_2 ... c_p] + t * a
    du[2p+1:end] = vec(fit.T * reshape(u[2p+1:end], p, p) + fit.t * u[1:p]')
end

function ode_censored(_, u, fit, du)
    p = length(fit.π)

    # da = a' * T (where the first p components of u are 'a')
    du[1:p] = u[1:p]' * fit.T

    # db = T * b (where the next p components of u are 'b')
    du[p+1:2p] = fit.T * u[p+1:2p]

    ## dc = T * [d_1 d_2 ... d_p] + [a_1*ones(p) a_2*ones(p) ... a_p*ones(p)]
    du[2p+1:end] = vec(fit.T * reshape(u[2p+1:end], p, p) .+ u[1:p]')
    println(du[2p+1:end])

    for i = 1:p
        du[(i+1)*p+1:(i+2)*p] = fit.T*u[(i+1)*p+1:(i+2)*p] + u[i]
    end

    println(du[2p+1:end])
end


function loglikelihood(s, fit)
    ll = 0.0

    if length(s.obs) > 0
        for v = 1:length(s.obs)
            ll += s.obsWeights[v] * log(fit.π' * expm(fit.T * s.obs[v]) * fit.t)
        end
    end

    # THIS SHOULD BE CHECKED! SURELY SOMETHING IS MISSING"
    if length(s.cens) > 0
        for v = 1:length(s.obs)
            ll += s.censWeights[v] * log(fit.π' * expm(fit.T * s.cens[v]) * fit.t)
        end
    end

    if length(s.ints) > 0
        error("Haven't implemented interval-censoring yet")
    end

    return ll
end

function parse_settings(settings_filename)
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
    settings = Dict()
    open(settings_filename, "r") do f
        dicttxt = readstring(f)
        settings = JSON.parse(dicttxt)
    end
    # I'd like to use the following, but it doesn't close the file properly.
    #settings = JSON.parsefile(settings_filename)

    name = get(settings, "Name", basename(settings_filename)[1:end-5])
    p = get(settings, "NumberPhases", 15)
    ph_structure = get(settings, "Structure", p < 20 ? "General" : "Coxian")
    continueFit = get(settings, "ContinuePreviousFit", true)
    num_iter = get(settings, "NumberIterations", 1_000)

    # Set the seed for the random number generation if requested.
    if haskey(settings, "RandomSeed")
        srand(settings["RandomSeed"])
    else
        srand(1337)
    end

    # Fill in the default values for the sample.
    s = settings["Sample"]

    obs = haskey(s, "Uncensored") ? s["Uncensored"]["Observations"] : []
    cens = haskey(s, "RightCensored") ? s["RightCensored"]["Cutoffs"] : []
    ints = haskey(s, "IntervalCensored") ? s["IntervalCensored"]["Intervals"] : []

    # Parse the list of intervals as a 2d array.
    ints = hcat(ints...)

    # Set the weights to 1 if not specified.
    obsWeights = length(obs) > 0 && haskey(s["Uncensored"], "Weights") ? s["Uncensored"]["Weights"] : ones(length(obs))
    censWeights = length(cens) > 0 && haskey(s["RightCensored"], "Weights") ? s["RightCensored"]["Weights"] : ones(length(cens))
    intsWeights = length(ints) > 0 && haskey(s["IntervalCensored"], "Weights") ? s["IntervalCensored"]["Weights"] : ones(length(ints))

    s = Sample(obs, obsWeights, cens, censWeights, ints, intsWeights)

    return (name, p, ph_structure, continueFit, num_iter, s)
end

function initial_phasetype(name, p, ph_structure, continueFit, s)
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
            T_legal = Tridiagonal(zeros(p-1), ones(p), ones(p-1)) .> 0
        else
            error("Nothing implemented for phase-type structure $ph_structure")
        end

        # Create a structure using standard uniforms.
        t = rand(p)

        π = rand(p)
        π[.~π_legal] = 0
        π /= sum(π)

        T = rand(p, p)
        T[.~T_legal] = 0
        T -= diagm(T*ones(p) + t)

        # Rescale t and T using the same scaling as in the EMPHT.c program.
        if length(s.obs) > min(length(s.cens), length(s.ints))
            scalefactor = median(s.obs)
        elseif length(s.ints) > length(s.cens)
            scalefactor = median(s.ints[2,:])
        else
            scalefactor = median(s.cens)
        end

        t *= p / scalefactor
        T *= p / scalefactor
    end

    return PhaseType(π, T, t)
end

function save_progress(name, p, s, fit, plotDens, plotMax)
    ll = loglikelihood(s, fit)
    println("loglikelihood $ll")

    open(string(name, "_loglikelihood.csv"), "a") do f
        write(f, "$ll\n")
    end

    writedlm(string(name, "_fit.csv"), [fit.π fit.T])

    if plotDens
        d(y) = fit.π' * expm(fit.T * y) * fit.t
        xVals = linspace(0, plotMax, 100)
        yVals = d.(xVals)
        fig = plot!(xVals, yVals)
        png(string(name, "_fit"))
        return fig
    end

    #println("Mean of fit is $(-fit.π' * inv(fit.T) * ones(p))")
end

function conditional_on_obs(s, p, fit, Bs, Zs, Ns)
    # Setup initial conditions.
    u0 = zeros(p*(p+2)); u0[1:p] = fit.π; u0[p+1:2p] = fit.t

    # Run the ODE solver.
    pf = ParameterizedFunction(ode_observations, fit)
    prob = ODEProblem(pf, u0, (0.0, 1.05*maximum(s.obs)))
    sol = solve(prob, BS3()) #BS3() tstops , saveat=s.obs

    for v = 1:length(s.obs)

        weight = s.obsWeights[v]
        u = sol(s.obs[v])

        a = u[1:p]; b = u[p+1:2p]
        π_by_b = fit.π .* b; denom = sum(π_by_b)

        Bs[:] = Bs[:] + weight * π_by_b / denom
        for i = 1:p
            c_i = u[(i+1)*p+1:(i+2)*p]
            Zs[i] = Zs[i] + weight * c_i[i] ./ denom
            Ns[i,end] = Ns[i,end] + weight * fit.t[i] * a[i] / denom

            for j = 1:p
                if i==j
                    continue
                end
                Ns[i,j] = Ns[i,j] + weight * fit.T[i,j] * c_i[j] / denom
            end
        end
    end
end

function conditional_on_cens(s, p, fit, Bs, Zs, Ns)
    # Setup initial conditions.
    u0 = zeros(p*(p+2)); u0[1:p] = fit.π; u0[p+1:2p] = 1

    # Run the ODE solver.
    pf = ParameterizedFunction(ode_censored, fit)
    prob = ODEProblem(pf, u0, (0.0, 1.1*maximum(s.cens)))
    sol = solve(prob, BS3(), saveat=s.cens) #tstops # BS3()

    for v = 1:length(s.cens)
        weight = s.censWeights[v]
        u = sol(s.cens[v])

        a = u[1:p]; h = u[p+1:2p]
        π_by_h = fit.π .* h; denom = sum(π_by_h)

        Bs[:] = Bs[:] + weight * π_by_h / denom
        for i = 1:p
            d_i = u[(i+1)*p+1:(i+2)*p]
            Zs[i] = Zs[i] + weight * d_i[i] ./ denom

            for j = 1:p
                if i==j
                    continue
                end
                Ns[i,j] = Ns[i,j] + weight * fit.T[i,j] * d_i[j] / denom
            end
        end
    end
end

function em(settings_filename)
    # Read in details for the fit from the settings file.
    name, p, ph_structure, continueFit, num_iter, s = parse_settings(settings_filename)
    println("name, p, ph_structure, continueFit, num_iter = $((name, p, ph_structure, continueFit, num_iter))")

    # Count the total of all weights.
    sumOfWeights = sum(s.obsWeights) + sum(s.censWeights) + sum(s.intsWeights)

    # Find the largest of the different samples to set appropriate plot size.
    plotMax = 1.1 * mapreduce(l -> length(l) > 0 ? maximum(l) : 0, max, (s.obs, s.cens, s.ints))

    # Construct the initial phase-type fit for the EM algorithm.
    fit = initial_phasetype(name, p, ph_structure, continueFit, s)

    println("Initial loglikelihood = $(loglikelihood(s, fit))")

    fig = plot()
    for iter = 1:num_iter
        if iter == 1 || iter % 25 == 0 # iter % ceil(num_iter / 10) == 0
            println("Starting iteration $iter")
        end

        ##  The expectation step!
        Bs = zeros(p); Zs = zeros(p); Ns = zeros(p, p+1)

        if length(s.obs) > 0
            conditional_on_obs(s, p, fit, Bs, Zs, Ns)
        end

        if length(s.cens) > 0
            conditional_on_cens(s, p, fit, Bs, Zs, Ns)
        end

        if length(s.ints) > 0
            error("Haven't implemented interval-censoring yet")
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
        t_next = -T_next * ones(p)

        fit = PhaseType(π_next, T_next, t_next)

        # Plot each iteration at the beginning
        if iter == num_iter
            fig = save_progress(name, p, s, fit, true, plotMax)
        elseif true || iter % 25 == 0 # iter %  ceil(num_iter / 10) == 0
            save_progress(name, p, s, fit, true, plotMax)
        end
    end

    fig
end


@time em("example1-sample.json")
