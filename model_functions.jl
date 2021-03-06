## npt-saliva-testing
## Affan Shoukat, 2020 
## Center for Infectious Disease Modelling and Analysis 

# activate the folder environment to pull in the correct versions of the packages used
using Pkg
Pkg.activate(".")
Pkg.status()

using GLPK
using DataFrames 
using CSV 
using LsqFit
using Distributions
using OffsetArrays
using Optim
using Gnuplot
using DelimitedFiles
using Parameters
using Random
Gnuplot.options.verbose=false
pwd()
#println(Gnuplot.gpversion())


# read global data files 
const ll = CSV.File("data/likelihoods.csv") |> DataFrame! ; 
const TVAL_OFFSET = -15:40 # run time for the infection

# first column is digitized from https://www.ncbi.nlm.nih.gov/pmc/articles/PMC7151472/
# second column is digitized from https://faseb.onlinelibrary.wiley.com/doi/10.1096/fj.202001700RR
const sens_csv = CSV.File("data/sensitivity.csv", header=true) |> DataFrame! ;

@with_kw mutable struct ModelParams
    test_type::Symbol = :np
    inf_type::Symbol = :all
    firsttest::Int16 = 50
    testfreq::Int16 = 7
    exp_days::Int32 = 100
end

## read the likelihood file from the Ashcroft paper. 
function get_infectivity_curve(t, shape, rate, shift)
    dist = Gamma(shape, 1/rate)    
    pdfvals = pdf.(dist, t .+ shift) 
end
 
function sample_ashcroft()
    # return shape, rate, shift from ashcroft likelihood table
    N = nrow(ll)
    rr = rand(1:N)  
    shape, rate, shift = ll[rr, :].shape, ll[rr, :].rate, ll[rr, :].shift 
end

function get_ashcroft_curve(tvals)    
    randcurves = hcat([get_infectivity_curve(tvals, sample_ashcroft()...) for _=1:1000]...)
    ymeans_ashcroft = dropdims(mean(randcurves, dims=2), dims=2) 
    ylows_ashcroft = quantile.(eachrow(randcurves), [0.025])
    yhighs_ashcroft = quantile.(eachrow(randcurves), [0.975])
    return OffsetArray(ymeans_ashcroft, tvals), OffsetArray(ylows_ashcroft, tvals), OffsetArray(yhighs_ashcroft, tvals)
end

function get_nature_curve(tvals)
    shrash =  20.516508, 1.592124, 12.272481    # parameters from the corrected nature paper
    ymeans_nature = get_infectivity_curve(tvals, shrash...) 
    ymeans_offset = OffsetArray(ymeans_nature, tvals)
end

function get_sensitivity_data()
    # loads the sensitivity data from the const loaded DataFrame
    sens::Array{Float64, 1}  = Int64.(sens_csv.harvard[1:26])./100  # harvard data includes day 0 as symptom onset + 25 days of post symptoms
    sens_off = OffsetArray(sens, 0:25) # offset the array for easier indexing in fitting process. has nothing to do with TVAL_OFFSET
    return sens_off
end

# gompertz function
function gompertz(shift_value, tvals)
    xvals = tvals[1]:(tvals[end] + shift_value - 1)
     _gmp = @. exp(-1*exp((-xvals)))
    gmp = [_gmp[i+shift_value - 1] for i = 1:length(tvals)]
    shift_gmp = OffsetArray(gmp, tvals)
end

@inline function infect_start()
    # returns the start of infectiousness period BEFORE SYMPTOM ONSET 
    # so the interpretation here is a 'negative number' 
    lat_dist = truncated(LogNormal(1.434065, 0.6612), 3, 15)
    inc_period = Int.(round.(rand(lat_dist)))
    return inc_period .- 1 #infectiousness starts one day after incubation
end


function fitted_curves()
    # this gives us the sensitivity curves for NP tests
    # returns 13 sensitivity curves, one for each value of the incubation period from 2 to 13. 
    sens = get_sensitivity_data()
    ymeans_offset = get_nature_curve(-15:40)
    # define function to the minimized
    f_min(p, ngmp) = sum((sens[0:25] .- p[3].*ymeans_offset[0:25].^p[1] ./ (ymeans_offset[0:25].^p[1] .+ p[2]) .* ngmp[0:25]).^2)
    @. f(x, p, ngmp) = (p[3]*x^p[1]  / (x^p[1] + p[2])) *  ngmp
    dd = Dict{Int64, OffsetArray{Float64,1,Array{Float64,1}}}()
    for inf = 2:14 # since infectiousness can only start between 2 to 14 days (see `infect_start`) 
        ngmp = gompertz(inf, -15:40)     ## values must match TVAL_OFFSET
        optz = optimize(p -> f_min(p, ngmp), [1, 0.5, 0.5]; autodiff = :forward)
        p0 = Optim.minimizer(optz)
        #println(typeof(f(ymeans_offset, p0, ngmp)))
        #println(p0)
        dd[inf] = f(ymeans_offset, p0, ngmp)
    end
    return dd
end

function get_sensitivity(inf_start)
    ## this function not used anymore. 
    ## this was flawed as we were fitting everytime inf_start was sampled
    ## since there are only 14 curves, we don't need to sample 10000 times for the same fits
    sens = get_sensitivity_data()
    ngmp = gompertz(inf_start, -15:40)     ## values must match TVAL_OFFSET
    ymeans_offset = get_nature_curve(-15:40)   
   
    # define function to the minimized
    f_min(p) = sum((sens[0:25] .- p[3].*ymeans_offset[0:25].^p[1] ./ (ymeans_offset[0:25].^p[1] .+ p[2]) .* ngmp[0:25]).^2)
    # define the modified hill function
    @. f(x, p) = (p[3]*x^p[1]  / (x^p[1] + p[2])) *  ngmp
    optz = optimize(f_min, [1, 0.5, 0.5]; autodiff = :forward)
    p0 = Optim.minimizer(optz)
    #p0 = [0.5, 0.5, 0.5]
    println(p0)
    myhill = f(ymeans_offset, p0)
    return myhill
end

function run_scenario(mp::ModelParams, sens_curves)
    @unpack test_type, inf_type, firsttest, testfreq, exp_days = mp
    #println(mp)
    dayoftest = collect(firsttest:testfreq:exp_days)
    positivity_alltests = zeros(Float64, exp_days)   # vectors to store the results
    positivity_firsttest = zeros(Float64, exp_days) 
    #positivity_sectest = zeros(Float64, exp_days) 
    
    for t = 1:exp_days 
        inf_start = infect_start() # need a negative here for proper indexing
        sensitivity = sens_curves[inf_start] # get the fitted sensitivity curve for this infectiousness period 
        maxvalue =  maximum(sensitivity) # get the maximum value for normalizing purpose
        if test_type == :sal 
            sensitivity .= sensitivity ./ maxvalue # for saliva, normalize the curve.
        end
        test_over_days = length(-inf_start:15) # only 15 days post symptom onset        
        possible_tests = filter(x -> x > t && x < test_over_days + t , dayoftest)
        rel_idx = -inf_start .+ (possible_tests .- t)
        if inf_type == :pre
            rel_idx = filter(x -> x < 0, rel_idx)
        end
        
        # positivity all tests
        test_sens = map(rel_idx) do x 
            infect = sensitivity[x] 
            if test_type == :sal 
                infect = infect * rand(Uniform(0.70, 0.97)) # saliva is done based on a normalized curve of np 
            end
            infect
        end

        any_l = length(test_sens) # don't really need this check anymore since length(possible_tests) > 0 in this if branch
        if any_l > 0 
            positivity_firsttest[t] = test_sens[1] # just get the first one for now.       
            tot_fails = 1 .- test_sens            
            prob_success = any_l > 0.0 ? 1 - reduce(*, tot_fails) : 0.0
            positivity_alltests[t] = prob_success
        end       
    end
    return (alltests = positivity_alltests, firsttest = positivity_firsttest)
end


function run_sims()
    #Random.seed!(2746)
    println("starting simulations (sens already fitted)...")
    println("using threads: $(Threads.nthreads())")
    N = 10000 
    experiment_days = 150
    sens_curves = fitted_curves()
    mp = ModelParams()
    freq = (2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14)
    #freq = (2, 4, 5,  7, 8, 9, 14)
    st = (:np, :sal)
    tg = (:inf, :pre)    
    for (t, g, f) in Iterators.product(st, tg, freq)        
        mp.exp_days = experiment_days
        mp.firsttest = 28 # for the "first test" scenario, changing this to 28 or any other arbitrary number shouldn't affect results. 
        mp.testfreq = f
        mp.test_type = t 
        mp.inf_type = g
           
        results_alltests = zeros(Float64, experiment_days, N)
        results_firsttests = zeros(Float64, experiment_days, N)
        
        Threads.@threads for i = 1:N
            res = run_scenario(mp, sens_curves)
            results_alltests[:, i] = res.alltests
            results_firsttests[:, i] = res.firsttest
        end
        
        firstdays = vec(results_firsttests[1:(mp.firsttest - 1), :])
        mfd = mean(firstdays[findall(x -> x > 0, firstdays)])
        #mfd = mean(firstdays)
        println("$(f)days_$(g), $t mean: $(mean(mfd))")
        #println("$(f)days_$(g), $t")
        writedlm("/data/optimal_testing/all_$(t)_$(f)days_$(g).csv", results_alltests)
        #writedlm("/data/optimal_testing/first_$(t)_$(f)days_$(g).csv", results_firsttests)
    end
end



### basic script testing / single scenario. 

# N = 1000
# experiment_days = 60
# freq_of_test = 3
# window = freq_of_test:(freq_of_test*4)
# meanwindow = freq_of_test:(freq_of_test*4 - 1)
# results = zeros(Float64, experiment_days, N)
# for i = 1:N
#     results[:, i] = next_tests_probs(:sal, :all, freq_of_test, experiment_days)
# end
# totals = mean(results, dims=2)
# # totals_filtered = totals[2*freq_of_test:(experiment_days - 2*freq_of_test)]
# plot(totals[window], seriestype=:bar)
# mean(totals[meanwindow])
# quantile(totals[meanwindow], [0.025, 0.975])
# mean(totals_filtered)
# histogram(mean(results[:, 1:150], dims=1)')

