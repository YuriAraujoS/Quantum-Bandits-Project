include("enviroment.jl")
include("policy.jl")
using Plots

n = 100000
k = 50
bandit = create_bandit(k)

rew_greedy, opt_greedy = run_test(bandit, n, (b, est, t) -> policy_epsilon_greedy(b, est, 0.1))

rew_decay, opt_decay = run_test(bandit, n, (b, est, t) -> policy_epsilon_decay(b, est, 0.05, n, t))

acc_greedy = [sum(opt_greedy[1:t]) / t * 100 for t in 1:n]
acc_decay = [sum(opt_decay[1:t]) / t * 100 for t in 1:n]
plot(1:n, acc_greedy, label="Epsilon Fixo (0.1)", lw=2)
plot!(1:n, acc_decay, label="Epsilon Decay", lw=2)
xlabel!("Passos (t)")
ylabel!("% Escolha da Melhor Ação Acumulada")