## npt-saliva-testing
#using JuMP, Ipopt
using GLPK
using DataFrames 
using CSV 
using LsqFit
using Distributions
using OffsetArrays
using Optim
using Gnuplot
Gnuplot.options.verbose=false
pwd()
println(Gnuplot.gpversion())

const ll = CSV.File("data/likelihoods.csv") |> DataFrame! ; 
const sens_csv = CSV.File("data/sensitivity.csv", header=true) |> DataFrame! ;
const TVAL_OFFSET = -15:40 # run time for the infection.

## read the likelihood file from the Ashcroft paper. 
function get_infectivity_curve(t, shape, rate, shift)
    dist = Gamma(shape, 1/rate)    
    pdfvals = pdf.(dist, t .+ shift) 
end

# we don't really use ashcroft 
function sample_ashcroft()
    # return shape, rate, shift from ashcroft's likelihood
    N = nrow(ll)
    rr = rand(1:N)  
    shape, rate, shift = ll[rr, :].shape, ll[rr, :].rate, ll[rr, :].shift 
end

function get_ashcroft_curve(tvals)    
    #tvals = -15:25 ## 25 since the harvard sensitivity data only has 26 points
    randcurves = hcat([get_infectivity_curve(tvals, sample_ashcroft()...) for _=1:1000]...)
    ymeans_ashcroft = dropdims(mean(randcurves, dims=2), dims=2) 
    ylows_ashcroft = quantile.(eachrow(randcurves), [0.025])
    yhighs_ashcroft = quantile.(eachrow(randcurves), [0.975])
    return ymeans_ashcroft, ylows_ashcroft, yhighs_ashcroft
end

function get_nature_curve(tvals)
    #tvals = -15:25 ## 25 since the harvard sensitivity data only has 26 points
    shrash =  20.516508, 1.592124, 12.272481    # nature.
    ymeans_nature = get_infectivity_curve(tvals, shrash...) 
    ymeans_offset = OffsetArray(ymeans_nature, tvals)
end

function get_sensitivity_data()
    # loads the sensitivity data from the const DataFrame
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
    lat_dist = truncated(LogNormal(1.434065, 0.6612), 3, 15)
    inc_period = Int.(round.(rand(lat_dist)))
    return inc_period .- 1 #infectiousness starts one day after incubation
end

function get_sensitivity(inf_start)
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
    #println(p0)
    myhill = f(ymeans_offset, p0)
    return myhill
end

function plot_sensitivity(inf_start)
    myhill = get_sensitivity(4) # plot sensitivity curve with infectiouness starting at i days
    println("maximum: $(maximum(myhill))")
        
    saliva_curves = hcat([rand(Uniform(0.70, 0.97)) * myhill / maximum(myhill) for _ = 1:1000]...)
    saliva_mean = dropdims(mean(saliva_curves, dims=2), dims=2)
    slows = quantile.(eachrow(saliva_curves), [0.025])
    shighs = quantile.(eachrow(saliva_curves), [0.975])
    
    @gp "set term svg enhanced standalone mouse size 800,400" ## use jupyter
    @gp :- tit="Fitted Hill Curve" key="opaque" 
    #@gp :- TVAL_OFFSET ymeans_offset[TVAL_OFFSET] "with lines title 'rel infectivity'"
    @gp :- TVAL_OFFSET slows shighs "with filledcu title 'quantile' lw 2 lc rgb '#a1d99b' "
    @gp :- TVAL_OFFSET saliva_mean "with lines title 'saliva mean' lc rgb 'black' lw 3"
    @gp :- TVAL_OFFSET myhill[TVAL_OFFSET] "with boxes title 'npt' lc rgb 'blue'"
    @gp :- 0:25 get_sensitivity_data()[0:25] "with points title 'sensitivity' pt 7 pointsize 0.65 lc rgb 'orange'"
end

function next_tests_probs(test_type=:np, inf_type=:all, testfreq = 7, exp_days = 30)
    #test_type=:np; inf_type=:all; testfreq = 5; exp_days = 20
    dayoftest = collect(testfreq:testfreq:exp_days)
    days_positive = zeros(Float64, exp_days)   

    for t = 1:exp_days
        inf_start = infect_start()
        sensitivity = get_sensitivity(inf_start)
        maxvalue =  maximum(sensitivity)
        if test_type == :sal 
            sensitivity .= sensitivity ./ maxvalue # for saliva, normalize the curve.
        end
        
        total_infect_days = length(sensitivity) ## get the total infectiousness period 
        possible_tests = filter(x -> x > t && x < total_infect_days + t , dayoftest)

        # get the relative_infectivity index for the offset array
        rel_idx = -15 .+ (possible_tests .- t)
        if inf_type == :pre
            rel_idx = filter(x -> x < 0, rel_idx)
        end

         # see what happens on the day of the test and record that for t. 
        tot_fails = map(rel_idx) do x 
            infect = sensitivity[x]
            if test_type == :sal 
                infect = infect * rand(Uniform(0.70, 0.97)) # saliva is done based on a normalized curve of np 
            end
            failure = 1 - infect
        end
        any_l = length(tot_fails)
        prob_success = any_l > 0.0 ? 1 - reduce(*, tot_fails) : 0.0
        days_positive[t] = prob_success
    end
    return days_positive    
end

function run_sims()
    println("starting simulations...")
    println("using threads: $(Threads.nthreads())")
    N = 10
    experiment_days = 20 
    freq = (7, 14 )
    st = (:np, :sal)
    tg = (:all, :pre)    
    for (t, g, f) in Iterators.product(st, tg, freq)
        fname = ("/data/optimal_testing/$(t)_$(f)days_$(g).csv")
        println(fname)  
        freq_of_test = f       
        results = zeros(Float64, experiment_days, N)
        Threads.@threads for i = 1:N
            results[:, i] = next_tests_probs(t, g, freq_of_test, experiment_days)
        end
        # totals = sum(results, dims=2)./N
        # # plot(totals, seriestype=:bar)
        # # mean(totals)
        #writedlm(fname, results)
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

