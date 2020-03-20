include("A2_src.jl")
# using Revise # lets you change A2funcs without restarting julia!
# includet("A2_src.jl")
using Plots
using Statistics: mean
using Zygote
using Test
using Logging
using .A2funcs: log1pexp # log(1 + exp(x)) stable
using .A2funcs: factorized_gaussian_log_density
using .A2funcs: skillcontour!
using .A2funcs: plot_line_equal_skill!

function log_prior(zs)
  return factorized_gaussian_log_density(0, 0, zs)
end

function logp_a_beats_b(za,zb)
  return -(log1pexp.(-(za .- zb)))
end


function all_games_log_likelihood(zs,games)
  zs_a = zs[games[:, 1], :]
  zs_b = zs[games[:, 2], :]
  likelihoods = logp_a_beats_b(zs_a, zs_b)
  return sum(likelihoods, dims = 1)
end

function joint_log_density(zs,games)
  return log_prior(zs) .+ all_games_log_likelihood(zs,games)
end

@testset "Test shapes of batches for likelihoods" begin
  B = 15 # number of elements in batch
  N = 4 # Total Number of Players
  test_zs = randn(4,15)
  test_games = [1 2; 3 1; 4 2] # 1 beat 2, 3 beat 1, 4 beat 2
  @test size(test_zs) == (N,B)
  #batch of priors
  @test size(log_prior(test_zs)) == (1,B)
  # loglikelihood of p1 beat p2 for first sample in batch
  @test size(logp_a_beats_b(test_zs[1,1],test_zs[2,1])) == ()
  # loglikelihood of p1 beat p2 broadcasted over whole batch
  @test size(logp_a_beats_b.(test_zs[1,:],test_zs[2,:])) == (B,)
  # batch loglikelihood for evidence
  @test size(all_games_log_likelihood(test_zs,test_games)) == (1,B)
  # batch loglikelihood under joint of evidence and prior
  @test size(joint_log_density(test_zs,test_games)) == (1,B)
end

# Convenience function for producing toy games between two players.
two_player_toy_games(p1_wins, p2_wins) = vcat([repeat([1,2]',p1_wins), repeat([2,1]',p2_wins)]...)

# Example for how to use contour plotting code
plot(title="Example Gaussian Contour Plot",
    xlabel = "Player 1 Skill",
    ylabel = "Player 2 Skill"
   )
example_gaussian(zs) = exp(factorized_gaussian_log_density([-1.,2.],[0.,0.5],zs))
skillcontour!(example_gaussian)#; l="example gaussian")
plot_line_equal_skill!()
savefig(joinpath("plots","example_gaussian.pdf"))


# TODO: plot prior contours
plot(title="Prior Contour Plot",
    xlabel = "Player 1 Skill",
    ylabel = "Player 2 Skill"
   )
prior(zs) = exp(log_prior(zs))
skillcontour!(prior)
plot_line_equal_skill!()
savefig(joinpath("plots","prior_contours.pdf"))

# TODO: plot likelihood contours
plot(title="Likelihood Contour Plot",
    xlabel = "Player 1 Skill",
    ylabel = "Player 2 Skill"
   )
games_1 = two_player_toy_games(1, 0)
likelihood(zs) = exp.(all_games_log_likelihood(zs, games_1))
skillcontour!(likelihood)
plot_line_equal_skill!()
savefig(joinpath("plots","likelihood_contours.pdf"))

# TODO: plot joint contours with player A winning 1 game
plot(title="Joint Contour Plot (A winning 1 game)",
    xlabel = "Player 1 Skill",
    ylabel = "Player 2 Skill"
   )
joint(zs) = exp.(joint_log_density(zs, games_1))
skillcontour!(joint)
plot_line_equal_skill!()
savefig(joinpath("plots","joint_contours_1.pdf"))

# TODO: plot joint contours with player A winning 10 games
plot(title="Joint Contour Plot (A winning 10 game)",
    xlabel = "Player 1 Skill",
    ylabel = "Player 2 Skill"
   )
games_10 = two_player_toy_games(10, 0)
joint_10(zs) = exp.(joint_log_density(zs, games_10))
skillcontour!(joint_10)
plot_line_equal_skill!()
savefig(joinpath("plots","joint_contours_10.pdf"))

#TODO: plot joint contours with player A winning 10 games and player B winning 10 games
plot(title="Joint Contour Plot (A winning 10 game)",
    xlabel = "Player 1 Skill",
    ylabel = "Player 2 Skill"
   )
games_10_10 = two_player_toy_games(10, 10)
joint_10_10(zs) = exp.(joint_log_density(zs, games_10_10))
skillcontour!(joint_10_10)
plot_line_equal_skill!()
savefig(joinpath("plots","joint_contours_10_10.pdf"))


function elbo(params,logp,num_samples)
  mu = params[1]
  ls = params[2]
  s = randn(size(mu)[1], num_samples)
  samples = s .* exp.(ls) .+ mu
  logp_estimate = logp(samples)
  logq_estimate = factorized_gaussian_log_density(mu, ls, samples)
  return sum(logp_estimate .- logq_estimate) ./ num_samples# should return scalar (hint: average over batch)
end

# Conveinence function for taking gradients
function neg_toy_elbo(params; games = two_player_toy_games(1,0), num_samples = 100)
  # TODO: Write a function that takes parameters for q,
  # evidence as an array of game outcomes,
  # and returns the -elbo estimate with num_samples many samples from q
  logp(zs) = joint_log_density(zs,games)
  return -elbo(params,logp, num_samples)
end


# Toy game
num_players_toy = 2
toy_mu = [-2.,3.] # Initial mu, can initialize randomly!
toy_ls = [0.5,0.] # Initual log_sigma, can initialize randomly!
toy_params_init = (toy_mu, toy_ls)
neg_toy_elbo(toy_params_init)


function fit_toy_variational_dist(init_params, toy_evidence; num_itrs=200, lr= 1e-2, num_q_samples = 10)
  params_cur = init_params
  for i in 1:num_itrs
    #TODO: gradients of variational objective with respect to parameters
    grad_params = gradient(params ->
    neg_toy_elbo(params, games = toy_evidence, num_samples = num_q_samples)
    , params_cur)
    #TODO: update paramters with lr-sized step in descending gradient
    params_cur =  params_cur .- lr .* grad_params[1]
    neg_elbo = neg_toy_elbo(params_cur, games = toy_evidence, num_samples = num_q_samples)
    @info "elbo: $neg_elbo"#TODO: report the current elbbo during training
    # TODO: plot true posterior in red and variational in blue
    # hint: call 'display' on final plot to make it display during training
    plot();
    likelihood_tar(zs) = exp.(joint_log_density(zs, toy_evidence))
    likelihood_var(zs) = exp.(factorized_gaussian_log_density(params_cur[1], params_cur[2], zs))
    skillcontour!(likelihood_tar, colour=:red) #TODO: plot likelihood contours for target posterior
    plot_line_equal_skill!()
    display(skillcontour!(likelihood_var, colour=:blue)) #TODO: plot likelihood contours for variational posterior
  end
  return params_cur
end

#TODO: fit q with SVI observing player A winning 1 game
games_1 = two_player_toy_games(1,0)
init_params = (toy_mu, toy_ls)
fit_toy_variational_dist(init_params, games_1)
#TODO: save final posterior plots
savefig(joinpath("plots","fit_toy_variational_dist_1.pdf"))

#TODO: fit q with SVI observing player A winning 10 games
games_10 = two_player_toy_games(10,0)
init_params = (toy_mu, toy_ls)
fit_toy_variational_dist(init_params, games_10)
#TODO: save final posterior plots
savefig(joinpath("plots","fit_toy_variational_dist_10.pdf"))

#TODO: fit q with SVI observing player A winning 10 games and player B winning 10 games
games_10_10 = two_player_toy_games(10,10)
init_params = (toy_mu, toy_ls)
fit_toy_variational_dist(init_params, games_10_10)
#TODO: save final posterior plots
savefig(joinpath("plots","fit_toy_variational_dist_10_10.pdf"))

## Question 4
# Load the Data
using MAT
vars = matread("tennis_data.mat")
player_names = vars["W"]
tennis_games = Int.(vars["G"])
num_players = length(player_names)
print("Loaded data for $num_players players")


function fit_variational_dist(init_params, tennis_games; num_itrs=200, lr= 1e-2, num_q_samples = 10)
  params_cur = init_params
  for i in 1:num_itrs
    grad_params = gradient(params ->
    neg_toy_elbo(params, games = tennis_games, num_samples = num_q_samples)
    , params_cur)
    #TODO: gradients of variational objective wrt params
    params_cur = params_cur .- lr .* grad_params[1]
    #TODO: update parmaeters wite lr-sized steps in desending gradient direction
    neg_elbo = neg_toy_elbo(params_cur, games = tennis_games, num_samples = num_q_samples)
    @info "elbo: $neg_elbo" #TODO: report objective value with current parameters
  end
  return params_cur
end

# TODO: Initialize variational family
init_mu = randn(num_players)#random initialziation
init_log_sigma = randn(num_players)# random initialziation
init_params = (init_mu, init_log_sigma)

# Train variational distribution
trained_params = fit_variational_dist(init_params, tennis_games)


#TODO: 10 players with highest mean skill under variational model
#hint: use sortperm
perm = sortperm(trained_params[1])
plot(trained_params[1][perm], yerror=exp.(trained_params[2][perm]),
      xlabel = "Mean", ylabel = "Standard Error",
      title = "All Players' Skill Plot")
savefig(joinpath("plots","all_players'_skill.pdf"))

p_names = []
temp = reverse(perm)

for i in 1:10
  p_name = player_names[temp[i]]
  push!(p_names, p_name)
end
p_names

#TODO: joint posterior over "Roger-Federer" and ""Rafael-Nadal""
#hint: findall function to find the index of these players in player_names
roger = findfirst(x -> x == "Roger-Federer", player_names)[1]
rafael = findfirst(x -> x == "Rafael-Nadal", player_names)[1]
mu_fed_Nad = [trained_params[1][roger], trained_params[1][rafael]]
ls_fed_Nad = [trained_params[2][roger], trained_params[2][rafael]]

joint_pos(zs) = exp.(factorized_gaussian_log_density(mu_fed_Nad, ls_fed_Nad, zs))
plot(title="Joint Posterior over Roger Federer and Rafael Nadal",
    xlabel = "Roger Federer Skill",
    ylabel = "Rafael Nadal Skill"
   )
skillcontour!(joint_pos)
plot_line_equal_skill!()
savefig(joinpath("plots","Joint_Posterior_over_Roger_Federer_and_Rafael_Nadal.pdf"))

# 4f
using Distributions

function calculate_prob_a_higher_than_b(mu, ls)
  A = [1 -1; 0 1]
  sigma = exp.(ls)
  mu_ya = (A * mu)[1]
  sigma_ya = (A .* sigma .* A')[1][1]
  return 1 - cdf(Normal(mu_ya, sigma_ya), 0)
end

# 4g
# Roger-Federer's skill is higher than Rafael-Nadal's
calculate_prob_a_higher_than_b(mu_fed_Nad, ls_fed_Nad)

sample_Fed = rand(Normal(mu_fed_Nad[1], exp(ls_fed_Nad[1])), 10000)
sample_Nad = rand(Normal(mu_fed_Nad[2], exp(ls_fed_Nad[2])), 10000)

sum_F_N = sum(sample_Fed[i] > sample_Nad[i] for i in 1:10000)

prob_Fed_higher_than_Nad = sum_F_N / 10000

# 4h
lowest_mean = minimum(trained_params[1])
lowest_index = findfirst(x -> x == lowest_mean, trained_params[1])
lowest_ls = trained_params[2][lowest_index]
player_names[lowest_index]

mu_Fed_lowest = [mu_fed_Nad[1], lowest_mean]
ls_Fed_lowest = [ls_fed_Nad[1], lowest_ls]
calculate_prob_a_higher_than_b(mu_Fed_lowest, ls_Fed_lowest)

sample_lowest = rand(Normal(lowest_mean, exp(lowest_ls)), 10000)
sum_F_lowest = sum(sample_Fed[i] > sample_lowest[i] for i in 1:10000)
prob_Fed_higher_than_lowest = sum_F_lowest / 10000



# 4i

init_mu_10 = ones(num_players) * 10#random initialziation
init_log_sigma_10 = zeros(num_players)# random initialziation
init_params_10 = (init_mu_10, init_log_sigma_10)

# Train variational distribution
trained_params = fit_variational_dist(init_params_10, tennis_games)
