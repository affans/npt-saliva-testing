### optimal testing - covid19 project for seyed. 
### main run function is run_sims() 

using Statistics
using Plots 
using Distributions
using OffsetArrays
using StatsBase
using DelimitedFiles

# fixed curve for initial analysis... not used for results.
# const rel_infect = [0, 0, 0, 0, 0, 0, 0, 0.05, 0.13, 0.58, 1, 0.87, 0.74, 0.62, 0.49, 0.37, 0.24, 0.12, 0.09, 0.05, 0.03, 0, 0, 0, 0, 0]
# const off_rel_infect = OffsetArray(rel_infect, -10:15)

const rel_infect_lo = [0, 0, 0, 0, 0, 0, 0, 0, 0.4939, 0.8622, 0.6352, 0.4385, 0.2921, 0.1902, 0.1219, 0.0772, 0.0485, 0.0302, 0.0188, 0.0116, 0.0071, 0.0044, 0.0027, 0.0016, 0.001, 0.0006]
const rel_infect_hi = [0, 0, 0, 0, 0, 0, 0, 0.2106, 1, 0.9992, 0.8401, 0.6135, 0.4216, 0.28, 0.182, 0.1165, 0.0737, 0.0463, 0.0288, 0.0179, 0.011, 0.0068, 0.0042, 0.0025, 0.0015, 0.0009]
const off_rel_infect_lo = OffsetArray(rel_infect_lo, -10:15)
const off_rel_infect_hi = OffsetArray(rel_infect_hi, -10:15)

@inline @inbounds function get_a_curve()
    rel_infect_curve = similar(rel_infect_lo) # create an empty array with length/type of rel_infect_lo
    for i = 1:length(rel_infect_lo)
        if (rel_infect_lo[i] + rel_infect_hi[i]) == 0 
            rel_infect_curve[i] = 0
        else
            d = Uniform(rel_infect_lo[i], rel_infect_hi[i])
            rel_infect_curve[i] = rand(d)
        end
    end
    return rel_infect_curve
end

function test_get_a_curve()
    ## this function samples 1000 get_a_curves()
    p = plot(1:length(rel_infect_lo), rel_infect_lo, seriestype=:line, linewidth=2, seriescolor="blue")
    plot!(p,1:length(rel_infect_lo), rel_infect_lo, seriestype=:scatter, linewidth=2, seriescolor="blue")
    plot!(p, 1:length(rel_infect_hi), rel_infect_hi, seriestype=:line, linewidth=2, seriescolor="blue")
    plot!(p, 1:length(rel_infect_hi), rel_infect_hi, seriestype=:scatter, linewidth=2, seriescolor="blue")
    mycurves = [get_a_curve() for _=1:10]
    plot!(p, mycurves)
end

function get_lat_pre_days_fixed(exp_days)
    lat_dist = truncated(LogNormal(log(5.2), 0.1), 4, 7) # truncated between 4 and 7
    pre_dist = truncated(Gamma(1.058, 5/2.3), 0.8, 3)#truncated between 0.8 and 3
    lat_days = Int.(round.(rand(lat_dist, exp_days)))
    pre_days = Int.(round.(rand(pre_dist, exp_days)))
    return lat_days, pre_days
end

function get_lat_pre_days(exp_days)
    ## exp_days = total number of days in experiment
    lat_dist = truncated(LogNormal(log(5.2), 0.1), 2, 10) # truncated between 4 and 7
    pre_dist = Gamma(1.058, 5/2.3) ## truncate it later. 
    # sample exp_days of incubation 
    lat_days = Int.(round.(rand(lat_dist, exp_days)))
    # using the incubation days as the maximum number of presymp (substract 1 so that incubation != presymp)
    # set up truncated distribution for latent period
    pre_dist_trunc = truncated.(pre_dist, 0.8, (lat_days .- 1))
    # sample the presymptomatic days (note the broadcasting since we have n distirbutions now)
    pre_days = Int.(round.(rand.(pre_dist_trunc)))
    return lat_days, pre_days
end

function next_tests_probs(test_type=:np, inf_type=:all, testfreq = 7, exp_days = 7)
    #test_type=:np; inf_type=:all; testfreq = 2; exp_days = 14    
    dayoftest = Tuple([collect(testfreq:testfreq:exp_days)...])
    #println("days of test: $dayoftest")
    #new code
    lat_days, pre_days = get_lat_pre_days_fixed(exp_days) # get_lat_pre_days(exp_days)
    rel_infect_curve = OffsetArray(get_a_curve(), -10:15)
    #fixed code
    # lat_days, pre_days = get_lat_pre_days_fixed(exp_days) # get_lat_pre_days(exp_days)
    # rel_infect_curve = off_rel_infect
    
    # store results
    days_positive = zeros(Float64, exp_days)     
    pres_capture = zeros(Bool, exp_days)

    # go through each day of infection. 
    for t = 1:exp_days
        # get their incubation and presymptomatic period. pb < ib always.
        ib = -lat_days[t]  
        pb = -pre_days[t]
        # if person is infected today (i.e. t), see what tests they are eligible for in the future.
        # the only tests they are eligble for are the ones that are in their presymp+symp periods. 
        possible_tests = filter(x -> x > (t + abs(ib) + pb) && x < length(ib:15) + t , dayoftest)
        # get the relative_infectivity index for the offset array
        rel_idx = possible_tests .+ ib .- (t  + 1)
        if inf_type == :pre
            rel_idx = filter(x -> x < 0, rel_idx)
        end
        # see what happens on the day of the test and record that for t. 
        tot_fails = map(rel_idx) do x 
            infect = rel_infect_curve[x]
            test_sens = test_type == :np ? 1.0 : rand(Uniform(0.604, 0.966))
            trans = infect * test_sens
            failure = 1 - trans
        end
        any_l = length(tot_fails)
        prob_success = any_l > 0 ? 1 - reduce(*, tot_fails) : 0
        days_positive[t] = prob_success
    end
    return days_positive
end


function run_sims()
    freq = (2, 4, 5, 7, 14 )
    st = (:np, :sal)
    tg = (:all, :pre)
    for (t, g, f) in Iterators.product(st, tg, freq)
        fname = ("/data/optimal_testing/$(t)_$(f)days_$(g).csv")
        println(fname)  
        # old code
        N = 50000
        experiment_days = 180      
        freq_of_test = f       
        results = zeros(Float64, experiment_days, N)
        for i = 1:N
            results[:, i] = next_tests_probs(t, g, freq_of_test, experiment_days)
        end
        # totals = sum(results, dims=2)./N
        # # plot(totals, seriestype=:bar)
        # # mean(totals)
        writedlm(fname, results)
    end
end

run_sims()

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


