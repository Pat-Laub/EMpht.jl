using CSV
using JSON
using Tables

function loglikelihoodcensored(s::Sample, fit::PhaseType)
    ll = 0.0

    for k = 1:length(s.obs)
        ll += s.obsweight[k] * log(pdf(fit, s.obs[k]))
    end

    for k = 1:length(s.cens)
        ll += s.censweight[k] * log(1 - cdf(fit, s.cens[k]))
    end

    ll += log(factorial(big(size(s.int, 1))))
    p = fit.p; π = fit.π; T = fit.T;
    for k = 1:size(s.int, 1)
        ll_k = log(π' * (exp(T * s.int[k,1])-exp(T * s.int[k,2]))*ones(p))
        ll += s.intweight[k] * ll_k - log(factorial(big(s.intweight[k])))
    end

    ll
end

function initial_phasetype(name::String, p::Int, ph_structure::String,
        continueFit::Bool, s::Sample, verbose::Bool)
    # If there is a <Name>_phases.csv then read the data from there.
    phases_filename = string(name, "_fit.csv")

    if continueFit && isfile(phases_filename)
        if verbose
            println("Continuing fit in $phases_filename")
        end
        phases = CSV.read(phases_filename) |> Matrix
        π = phases[1:end, 1]
        T = phases[1:end, 2:end]
        if length(π) != p || size(T) != (p, p)
            error("Error reading $phases_filename, expecting $p phases")
        end
        t = -T * ones(p)

    else # Otherwise, make a random start for the matrix.
        if verbose
            println("Using a random starting value")
        end
        if ph_structure == "General"
            π_legal = trues(p)
            T_legal = trues(p, p)
        elseif ph_structure == "Coxian" || ph_structure == "CanonicalForm1"
            π_legal = 1:p .== 1
            T_legal = diagm(1 => ones(p-1)) .> 0
        elseif ph_structure == "GeneralisedCoxian"
            π_legal = trues(p)
            T_legal = diagm(1 => ones(p-1)) .> 0
        else
            error("Nothing implemented for phase-type structure $ph_structure")
        end

        # Create a structure using [0.1, 1] uniforms.
        t = (0.9 * rand(p) .+ 0.1)

        π = (0.9 * rand(p) .+ 0.1)
        π[.~π_legal] .= 0
        π /= sum(π)

        T = (0.9 * rand(p, p) .+ 0.1)
        T[.~T_legal] .= 0
        T -= diagm(0 => T*ones(p) + t)

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

        if ph_structure == "CanonicalForm1"
            # Convert the original Coxian phase-type into the canonical form.
            π, T, t = reverse_coxian(π, T, t)
            π, T, t = sort_into_canonical_form(π, T, t)
        end
    end

    PhaseType(π, T, t)
end

function save_progress(name::String, s::Sample, fit::PhaseType, start::DateTime)
    ll = loglikelihoodcensored(s, fit)

    if ~isempty(name)
        open(string(name, "_loglikelihood.csv"), "a") do f
            mins = (now() - start).value / 1000 / 60
            write(f, "$ll $(round(mins; digits=4))\n")
        end

        CSV.write(string(name, "_fit.csv"), [fit.π fit.T] |> Tables.table)
    end

    ll
end

function parse_settings(settings_filename::String, verbose=false)
    # Check the file exists.
    if ~isfile(settings_filename)
        error("Settings file $settings_filename not found.")
    end

    # Check the input file is a json file.
    if length(settings_filename) < 6 || settings_filename[end-4:end] != ".json"
        error("Require a settings file as 'filename.json'.")
    end

    # Read in the properties of this fit (e.g. number of phases, PH structure)
    if verbose
        println("Reading settings from $settings_filename")
    end
    settings = JSON.parsefile(settings_filename, use_mmap=false)

    name = get(settings, "Name", basename(settings_filename)[1:end-5])
    p = get(settings, "NumberPhases", 15)
    ph_structure = get(settings, "Structure", p < 20 ? "General" : "Coxian")
    continueFit = get(settings, "ContinuePreviousFit", false)
    max_iter = get(settings, "NumberIterations", 1_000)
    timeout = get(settings, "TimeOut", 30)

    # Set the seed for the random number generation if requested.
    if haskey(settings, "RandomSeed")
        Random.seed!(settings["RandomSeed"])
    else
        Random.seed!(1)
    end

    # Fill in the default values for the sample.
    s = settings["Sample"]

    obs = haskey(s, "Uncensored") ?
        Vector{Float64}(s["Uncensored"]["Observations"]) : Vector{Float64}()
    cens = haskey(s, "RightCensored") ?
        Vector{Float64}(s["RightCensored"]["Cutoffs"]) : Vector{Float64}()
    int = haskey(s, "IntervalCensored") ?
        Matrix{Float64}(hcat(s["IntervalCensored"]["Intervals"]...)') :
            Matrix{Float64}(undef, 0, 0)

    # Set the weight to 1 if not specified.
    obsweight = length(obs) > 0 && haskey(s["Uncensored"], "Weights") ?
        Vector{Float64}(s["Uncensored"]["Weights"]) : ones(length(obs))
    censweight = length(cens) > 0 && haskey(s["RightCensored"], "Weights") ?
        Vector{Float64}(s["RightCensored"]["Weights"]) : ones(length(cens))
    intweight = length(int) > 0 && haskey(s["IntervalCensored"], "Weights") ?
        Vector{Float64}(s["IntervalCensored"]["Weights"]) : ones(length(int))

    s = Sample(obs, obsweight, cens, censweight, int, intweight)

    (s, p, ph_structure, name, continueFit, max_iter, timeout)
end
