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
        cond = all(obs .>= 0) && all(obsweight .>= 0) && all(cens .>= 0) &&
                all(censweight .>= 0) && all(int .>= 0) && all(intweight .>= 0)
        if ~cond
            error("Require non-negativity of observations and weights")
        end

        new(obs, obsweight, cens, censweight, int, intweight)
    end
end

function Sample(;obs::Vector{Float64}=Vector{Float64}(),
        obsweight::Vector{Float64}=Vector{Float64}(),
        cens::Vector{Float64}=Vector{Float64}(),
        censweight::Vector{Float64}=Vector{Float64}(),
        int::Matrix{Float64}=Matrix{Float64}(undef, 0, 0),
        intweight::Vector{Float64}=Vector{Float64}())

    if ~isempty(obs) && isempty(obsweight)
        obsweight = ones(length(obs))
    end
    if ~isempty(cens) && isempty(censweight)
        censweight = ones(length(cens))
    end
    if ~isempty(int) && isempty(intweight)
        intweight = ones(length(int))
    end

    Sample(obs, obsweight, cens, censweight, int, intweight)
end
