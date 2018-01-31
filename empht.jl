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

function ode_observations!(_, u::AbstractArray{Float64}, fit::PhaseType, du::AbstractArray{Float64})
    p = fit.p
    u[1:end] = max.(u, 0)

    # da = a' * T (where the first p components of u are 'a')
    du[1:p] = transpose(u[1:p]) * fit.T

    # db = T * b (where the next p components of u are 'b')
    du[p+1:2p] = fit.T * u[p+1:2p]

    # dc = T * [c_1 c_2 ... c_p] + t * a
    du[2p+1:end] = vec(fit.T * reshape(u[2p+1:end], p, p) + fit.t * transpose(u[1:p]))
end

function ode_censored!(_, u::AbstractArray{Float64}, fit::PhaseType, du::AbstractArray{Float64})
    p = fit.p
    u[1:end] = max.(u, 0)

    # da = a' * T (where the first p components of u are 'a')
    du[1:p] = transpose(u[1:p]) * fit.T

    # db = T * b (where the next p components of u are 'b')
    du[p+1:2p] = fit.T * u[p+1:2p]

    # dc = T * [d_1 d_2 ... d_p] + [a_1*ones(p) a_2*ones(p) ... a_p*ones(p)]
    du[2p+1:end] = vec(fit.T * reshape(u[2p+1:end], p, p) .+ transpose(u[1:p]))
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
        cdf_upper = cdf(fit, s.int[k,2])
        cdf_lower = cdf(fit, s.int[k,1])
        ll += s.intweight[k] * log(cdf_upper - cdf_lower)
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

    # Set the seed for the random number generation if requested.
    if haskey(settings, "RandomSeed")
        srand(settings["RandomSeed"])
    else
        srand(1337)
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

    (name, p, ph_structure, continueFit, num_iter, s)
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
        if length(s.obs) > min(length(s.cens), length(s.int))
            scalefactor = median(s.obs)
        elseif length(s.int) > length(s.cens)
            scalefactor = median(s.int[2,:])
        else
            scalefactor = median(s.cens)
        end

        t *= p / scalefactor
        T *= p / scalefactor
    end

    PhaseType(π, T)
end

function save_progress(name::String, s::Sample, fit::PhaseType, plotDens::Bool, plotMax::Float64)
    ll = loglikelihoodcensored(s, fit)
    println("loglikelihood $ll")

    open(string(name, "_loglikelihood.csv"), "a") do f
        write(f, "$ll\n")
    end

    writedlm(string(name, "_fit.csv"), [fit.π fit.T])

    if plotDens
        xVals = linspace(0, plotMax, 100)
        yVals = pdf.(fit, xVals)
        fig = plot!(xVals, yVals)
        png(string(name, "_fit"))
        fig
    end
end

function ensure_positive!(u)
    if ~all(u .>= 0)
        if all(isapprox.(min.(u, 0), 0, atol=1e-4))
            #warn("ODE solutions slightly negative")
            u[1:end] = max.(u, 0)
        else
            error("All ODE solutions should be non-negative: got $u")
        end
    end
end

function conditional_on_obs!(s::Sample, fit::PhaseType, Bs::AbstractArray{Float64}, Zs::AbstractArray{Float64}, Ns::AbstractArray{Float64})
    # Setup initial conditions.
    p = fit.p
    u0 = zeros(p*(p+2)); u0[1:p] = fit.π; u0[p+1:2p] = fit.t

    # Run the ODE solver.
    pf = ParameterizedFunction(ode_observations!, fit)
    prob = ODEProblem(pf, u0, (0.0, 1.05*maximum(s.obs)))
    sol = solve(prob, BS3()) #BS3() tstops , saveat=s.obs

    for k = 1:length(s.obs)
        weight = s.obsweight[k]
        u = sol(s.obs[k])
        ensure_positive!(u)

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

function conditional_on_cens!(s::Sample, fit::PhaseType, Bs::AbstractArray{Float64}, Zs::AbstractArray{Float64}, Ns::AbstractArray{Float64})
    # Setup initial conditions.
    p = fit.p
    u0 = zeros(p*(p+2)); u0[1:p] = fit.π; u0[p+1:2p] = 1

    # Should tell the ODE solver to evaluate all right and interval censored points.
    allcens = vcat(s.cens, vec(s.int))

    # Run the ODE solver.
    pf = ParameterizedFunction(ode_censored!, fit)
    prob = ODEProblem(pf, u0, (0.0, 1.1*maximum(allcens)))
    sol = solve(prob, BS3(), saveat=allcens) #tstops # BS3()

    for k = 1:length(s.cens)
        weight = s.censweight[k]
        u = sol(s.cens[k])
        ensure_positive!(u)

        h = u[p+1:2p]
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

    for k = 1:size(s.int, 1)
        weight = s.intweight[k]
        left, right = s.int[k,:]

        u_left = sol(left); u_right = sol(right)
        ensure_positive!(u_left); ensure_positive!(u_right)

        #h_left = u_left[p+1:2p]; h_right = u_right[p+1:2p]
        Δh = u_right[p+1:2p] - u_left[p+1:2p]
        #a_left = u_left[1:p]; a_right = u_right[1:p]
        #g_left = (a_left - fit.π) * fit.Tinv; g_right = (a_right - fit.π) * fit.Tinv
        Δg = transpose(u_right[1:p] - u_left[1:p]) * fit.Tinv
        π_by_minusΔh = -fit.π .* Δh; denom = sum(π_by_minusΔh)
        Bs[:] = Bs[:] + weight * π_by_minusΔh / denom

        for i = 1:p
            Δd_i = u_right[(i+1)*p+1:(i+2)*p] - u_left[(i+1)*p+1:(i+2)*p]
            Zs[i] = Zs[i] + weight * (Δg[i] - Δd_i[i]) / denom ####### TO fix, goes negative
            Ns[i,end] = Ns[i,end] + weight * fit.t[i] * Δg[i] / denom

            for j = 1:p
                if i==j
                    continue
                end
                Ns[i,j] = Ns[i,j] + weight * fit.T[i,j] * (Δg[i] - Δd_i[j]) / denom
            end
        end
    end
end

function em(settings_filename::String)
    # Read in details for the fit from the settings file.
    name, p, ph_structure, continueFit, num_iter, s = parse_settings(settings_filename)
    println("name, p, ph_structure, continueFit, num_iter = $((name, p, ph_structure, continueFit, num_iter))")

    # Check we don't just have right-censored obs, since this blows things up.
    if length(s.obs) == 0 && length(s.cens) > 0 && length(s.int) == 0
        error("Can't just have right-censored observations!")
    end

    # Count the total of all weight.
    sumOfWeights = sum(s.obsweight) + sum(s.censweight) + sum(s.intweight)

    # Find the largest of the different samples to set appropriate plot size.
    plotMax = 1.1 * mapreduce(l -> length(l) > 0 ? maximum(l) : 0, max, (s.obs, s.cens, s.int))

    # Construct the initial phase-type fit for the EM algorithm.
    fit = initial_phasetype(name, p, ph_structure, continueFit, s)

    println("Initial loglikelihood = $(loglikelihoodcensored(s, fit))")

    fig = plot()
    for iter = 1:num_iter
        if iter == 1 || iter % 25 == 0 # iter % ceil(num_iter / 10) == 0
            println("Starting iteration $iter")
        end

        ##  The expectation step!
        Bs = zeros(p); Zs = zeros(p); Ns = zeros(p, p+1)

        if length(s.obs) > 0
            conditional_on_obs!(s, fit, Bs, Zs, Ns)
        end

        if length(s.cens) > 0 || length(s.int) > 0
            conditional_on_cens!(s, fit, Bs, Zs, Ns)
        end

        ## The maximisation step!
        π_next = Bs ./ sumOfWeights
        t_next = Ns[:,end] ./ Zs
        println("Ns[:,end] = $(Ns[:,end])")
        println("Zs = $Zs")
        println("t_next = $t_next")

        T_next = zeros(p,p)
        for i=1:p
            T_next[i,:] = Ns[i,1:end-1] ./ Zs[i]
            T_next[i,i] = -(t_next[i] + sum(T_next[i,:]))
        end

        # Remove any numerical instabilities.
        π_next = max.(π_next, 0)
        π_next /= sum(π_next)

        fit = PhaseType(π_next, T_next)

        # Plot each iteration at the beginning
        if iter == num_iter
            fig = save_progress(name, s, fit, true, plotMax)
        elseif true || iter % 25 == 0 # iter %  ceil(num_iter / 10) == 0
            save_progress(name, s, fit, false, plotMax)
        end
    end

    fig
end


@time em("example1-sample.json")
@time em("cexample1-sample.json")
@time em("example2-sample.json")
