using JSON
using OrdinaryDiffEq
using Plots

function phase_type_e_step(tUnused, u, chainParams, du)
    p = Int(sqrt(length(u) + 1) - 1)
    T = chainParams["T"]; t = chainParams["t"]

    # da = a' * T (where the first p components of u are 'a')
    du[1:p] = u[1:p]' * T

    # db = T * b (where the next p components of u are 'b')
    du[p+1:2p] = T * u[p+1:2p]

    ## dc = T * [c_1 c_2 ... c_p] + t * a
    du[2p+1:end] = vec(T * reshape(u[2p+1:end], p, p) + t * u[1:p]')
end

function loglikelihood(s, π, T, t)
    ll = 0.0

    if s["Uncensored"]["Count"] > 0
        for v = 1:s["Uncensored"]["Count"]
            ll += s["Uncensored"]["Weights"][v] * log(π' * expm(T * s["Uncensored"]["Observations"][v]) * t)
        end
    end

    if s["RightCensored"]["Count"] > 0
        error("Haven't implemented right-censoring yet")
    end

    if s["IntervalCensored"]["Count"] > 0
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
    get!(s, "Uncensored", Dict{String,Any}("Observations" => []))
    get!(s, "RightCensored", Dict{String,Any}("Cutoffs" => []))
    get!(s, "IntervalCensored", Dict{String,Any}("Intervals" => []))

    # Parse the list of intervals as a 2d array.
    s["IntervalCensored"]["Intervals"] = hcat(s["IntervalCensored"]["Intervals"]...)

    # Count the number of each kind of sample.
    s["Uncensored"]["Count"] = length(s["Uncensored"]["Observations"])
    s["RightCensored"]["Count"] = length(s["RightCensored"]["Cutoffs"])
    s["IntervalCensored"]["Count"] = length(s["IntervalCensored"]["Intervals"])

    # Set the weights to 1 if not specified.
    get!(s["Uncensored"], "Weights", ones(s["Uncensored"]["Count"]))
    get!(s["RightCensored"], "Weights", ones(s["RightCensored"]["Count"]))
    get!(s["IntervalCensored"], "Weights", ones(s["IntervalCensored"]["Count"]))

    # Count the total of all weights.
    s["SumOfWeights"] = sum(s["Uncensored"]["Weights"]) + sum(s["RightCensored"]["Weights"]) + sum(s["IntervalCensored"]["Weights"])

    # Find the largest of the different samples.
    maxWithZero(itr) = length(itr) > 0 ? maximum(itr) : 0
    s["MaxSample"] = mapreduce(maxWithZero, max, (s["Uncensored"]["Observations"], s["RightCensored"]["Cutoffs"], s["IntervalCensored"]["Intervals"]))

    return (name, p, ph_structure, continueFit, num_iter, s)
end

function initial_phasetype(name, p, ph_structure, continueFit, s)
    # If there is a <Name>_phases.csv then read the data from there.
    phases_filename = string(name, "_fit.csv")

    if continueFit && isfile(phases_filename)
        println("Continuing fit in $phases_filename")
        phases = readdlm(phases_filename)
        π_iter = phases[1:end, 1]
        T_iter = phases[1:end, 2:end]
        if length(π_iter) != p || size(T_iter) != (p, p)
            error("Error reading $phases_filename, expecting $p phases")
        end
        t_iter = -T_iter * ones(p)

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
        t_iter = rand(p)

        π_iter = rand(p)
        π_iter[.~π_legal] = 0
        π_iter /= sum(π_iter)

        T_iter = rand(p, p)
        T_iter[.~T_legal] = 0
        T_iter -= diagm(T_iter*ones(p) + t_iter)

        # Rescale t and T using the same scaling as in the EMPHT.c program.
        if s["Uncensored"]["Count"] > min(s["RightCensored"]["Count"], s["IntervalCensored"]["Count"])
            scalefactor = median(s["Uncensored"]["Observations"])
        elseif s["IntervalCensored"]["Count"] > s["RightCensored"]["Count"]
            scalefactor = median(s["IntervalCensored"]["Intervals"][2,:])
        else
            scalefactor = median(s["RightCensored"]["Cutoffs"])
        end

        t_iter *= p / scalefactor
        T_iter *= p / scalefactor
    end

    return (π_iter, T_iter, t_iter)
end

function save_progress(name, p, s, π_iter, T_iter, t_iter, plot_dens)
    ll = loglikelihood(s, π_iter, T_iter, t_iter)
    println("loglikelihood $ll")

    open(string(name, "_loglikelihood.csv"), "a") do f
        write(f, "$ll\n")
    end

    writedlm(string(name, "_fit.csv"), [π_iter T_iter])

    if plot_dens
        d(y) = π_iter' * expm(T_iter * y) * t_iter
        xVals = linspace(0, s["MaxSample"]*1.25, 100)
        yVals = d.(xVals)
        fig = plot!(xVals, yVals)
        png(string(name, "_fit"))
        return fig
    end

    #println("Mean of fit is $(-π_iter' * inv(T_iter) * ones(p))")
end

function em(settings_filename)
    # Read in details for the fit from the settings file.
    name, p, ph_structure, continueFit, num_iter, s = parse_settings(settings_filename)
    println("name, p, ph_structure, continueFit, num_iter = $((name, p, ph_structure, continueFit, num_iter))")

    # Construct the initial phase-type fit for the EM algorithm.
    π_iter, T_iter, t_iter = initial_phasetype(name, p, ph_structure, continueFit, s)

    fig = plot()
    for iter = 1:num_iter
        if iter == 1 || iter % 25 == 0 # iter % ceil(num_iter / 10) == 0
            println("Starting iteration $iter")
        end

        ##  The expectation step!
        Bs = zeros(p); Zs = zeros(p)
        Ns = zeros(p, p); NTerms = zeros(p)

        if s["Uncensored"]["Count"] > 0
            # Setup initial conditions.
            u0 = zeros(p*(p+2)); u0[1:p] = π_iter; u0[p+1:2p] = t_iter

            # Run the ODE solver.
            chainParams = Dict("T" => T_iter, "t" => t_iter)
            pf = ParameterizedFunction(phase_type_e_step, chainParams)
            prob = ODEProblem(pf, u0, (0.0, 1.1*maximum(s["Uncensored"]["Observations"])))
            sol = solve(prob, BS3(), saveat=s["Uncensored"]["Observations"]) #, saveat=s["Uncensored"]["Observations"]) #tstops # BS3()

            for v = 1:s["Uncensored"]["Count"]
                weight = s["Uncensored"]["Weights"][v]
                u_v = sol(s["Uncensored"]["Observations"][v])

                a_v = u_v[1:p]; b_v = u_v[p+1:2p]
                π_by_b = π_iter .* b_v; denom = sum(π_by_b)

                Bs += weight * π_by_b ./ denom
                for i = 1:p
                    c_v_i = u_v[(i+1)*p+1:(i+2)*p]
                    Zs[i] += weight * c_v_i[i] ./ denom
                    NTerms[i] += weight * t_iter[i] * a_v[i] / denom

                    for j = 1:p
                        if i==j
                            continue
                        end
                        Ns[i,j] += weight * T_iter[i,j] * c_v_i[j] / denom
                    end
                end
            end
        end

        if s["RightCensored"]["Count"] > 0
            error("Haven't implemented right-censoring yet")
        end

        if s["IntervalCensored"]["Count"] > 0
            error("Haven't implemented interval-censoring yet")
        end

        ## The maximisation step!
        π_iter = Bs ./ s["SumOfWeights"]
        t_iter = NTerms ./ Zs

        T_iter = zeros(p,p)
        for i=1:p
            T_iter[i,:] = Ns[i,:] ./ Zs[i]
            T_iter[i,i] = -(t_iter[i] + sum(T_iter[i,:]))
        end

        # Remove any numerical instabilities.
        π_iter = max.(π_iter, 0)
        π_iter /= sum(π_iter)
        t_iter = -T_iter * ones(p)

        # Plot each iteration at the beginning
        if iter == num_iter
            fig = save_progress(name, p, s, π_iter, T_iter, t_iter, true)
        elseif iter % 25 == 0 # iter %  ceil(num_iter / 10) == 0
            save_progress(name, p, s, π_iter, T_iter, t_iter, false)
        end
    end

    fig
end

gr()

em("example1-sample.json")
