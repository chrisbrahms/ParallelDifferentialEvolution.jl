module ParallelDifferentialEvolution
export diffevo, nullcb, logcb

import StatsBase
import Logging
import Statistics: mean, std
import Dates
import LatinHypercubeSampling

function nullcb(gen, x, im, f, con, etime)
end

function logcb(gen, x, im, f, con, etime)
    @info """Generation: $gen
    \tConvergence: $con
    \tElapsed time: $etime
    \tBest x: $(x[im])
    \tBest f: $(f[im])
    \tmean(f): $(mean(f))
    \tstd(f): $(std(f))"""
    flush(stdout)
    flush(stderr)
end

# fo is the objective function (must be @everywhere for distributed use) 
# we minimise the fitness
# bounds translates each parameter from [0, 1] to [min, max]
# d is dimensionality of problem
# F should be 0.4 < F < 1.0; small F speeds convergence but can cause premature convergence
# CR should be 0.1 < CR < 1.0; higher CR speeds convergence
# np should be between 5d and 10d, and must be at least 4
# see https://doi.org/10.1023/A:1008202821328
# rtol and atol set convergence tolerance: std(f) < (atol + rtol*abs(mean(f)))
# cb is a callback to use after eacg generation: defaulst to null, logcb prints stats
# fmap is the map to use to evaluate fitness: either `map` or `pmap`
# seeds can be a list of individuals to insert into the initial population
function diffevo(fo, d; F=0.8, CR=0.6, np=d*10,
                 maxiter=1000, rtol=1e-3, atol=1e-14, cb=nullcb, fmap=map,
                 seeds=nothing)
    @info "Making initial seed from latin hypercube..."
    plan, _ = LatinHypercubeSampling.LHCoptim(np, d, 1000)
    plan = LatinHypercubeSampling.scaleLHC(plan, repeat([(0.0, 1.0)], d))
    x = mapslices(x->[x], plan, dims=2)
    @info "...initial seed done."
    flush(stdout)
    flush(stderr)
    if !isnothing(seeds)
        for (i,seed) in enumerate(seeds)
            x[i] = seed
        end
    end
    f = fmap(fo, x)
    im = argmin(f)
    m = x[im]
    trials = similar(x)
    stime = Dates.now()
    gen = 1
    while gen <= maxiter
        for j in 1:np
            ii = [k for k in 1:np if k != j]
            a, b, c = x[StatsBase.sample(ii, 3, replace=false)]
            dither = F + rand()*(1 - F)
            mut = clamp.(a .+ dither .* (b .- c), 0.0, 1.0)
            cp = rand(d) .< CR
            if !any(cp)
                cp[rand(1:d)] = true
            end
            trials[j] = ifelse.(cp, mut, x[j])
        end
        tf = try
            fmap(fo, trials)
        catch e
            bt = catch_backtrace()
            msg = "Error in pmap:\n"*sprint(showerror, e, bt)
            @warn msg
            throw(e)
        end
        for j in 1:np
            if tf[j] < f[j]
                f[j] = tf[j]
                x[j] = trials[j]
                if f[j] < f[im]
                    im = j
                    m = trials[j]
                end
            end
        end
        global con = std(f) / (atol + rtol*abs(mean(f)))
        global etime = floor(Dates.now() - stime, Dates.Second)
        cb(gen, x, im, f, con, etime)
        if con < 1.0
           @info "converged on generation $gen"
           break
        end
        gen += 1
    end
    if con >= 1.0
        @info "maximum number of $gen iterations reached"
    end
    return (F=F, CR=CR, np=np, maxiter=maxiter, gen=gen, rtol=rtol, atol=atol,
            m=m, f=f, fm=f[im], etime=etime, x=x, im=im, con=con)
end

end
