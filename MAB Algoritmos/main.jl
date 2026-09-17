include("enviroment.jl")
include("policy.jl")
using Plots, Statistics

T = 250
R = 1000
t = 1:T

sigmas = [6.0, 3.0, 1.5]
mus = [1.0, 2.0, 3.0, 4.0, 5.0, 1.5, 2.5, 3.5, 4.5, 5.5, 0.5]

policies = [
    ("Epsilon Fixo (0.1)", (b, est, step) -> policy_epsilon_greedy(b, est, 0.1)),
    ("Epsilon Decay",      (b, est, step) -> policy_epsilon_decay(b, est, 0.05, T, step)),
    ("UCB (c=2.0)",        (b, est, step) -> policy_ucb(b, est, step, sqrt(2.0)))
]

# Lista com o Oráculo para os gráficos comparativos
policies_with_oracle = [
    ("Oráculo",            (b, est, step) -> policy_oracle(b, est, step)),
    policies...
]

# --- 1. Identificação da Ação Ótima ao Longo do Tempo ---
for sigma in sigmas
    bandit = [[mu, sigma] for mu in mus]
    
    p = plot(title="Identificação da Ação Ótima (σ = $(round(sigma, digits=3)))",
             xlabel="Rodada (t)",
             ylabel="Estado Médio [Par = Erro, Ímpar = Ótimo]",
             yticks=(0:5, ["0: Fixo Erro", "1: Fixo Ótimo", 
                           "2: Decay Erro", "3: Decay Ótimo", 
                           "4: UCB Erro",  "5: UCB Ótimo"]),
             ylims=(-0.5, 5.5),
             legend=:outerright,
             size=(850, 480),
             left_margin=8Plots.mm,
             bottom_margin=5Plots.mm)
    
    for (idx, (label_name, policy_fn)) in enumerate(policies)
        offset = 2.0 * (idx - 1)
        c = palette(:default)[idx]

        hit_matrix = zeros(Float64, R, T)
        for rep in 1:R
            _, is_optimal = run_test(bandit, T, policy_fn)
            hit_matrix[rep, :] = is_optimal
        end
        
        encoded_curve = offset .+ vec(mean(hit_matrix, dims=1))

        plot!(p, t, encoded_curve, label=label_name, color=c, lw=2)
        hline!(p, [offset + 0.5], linestyle=:dash, color=c, alpha=0.4, label="")
    end
    
    display(p)
    sleep(1.0)
end

# --- 2. Regret Acumulado Médio ---
for sigma in sigmas
    bandit = [[mu, sigma] for mu in mus]
    
    oracle_matrix = zeros(Float64, R, T)
    for rep in 1:R
        rewards, _ = run_test(bandit, T, policy_oracle)
        oracle_matrix[rep, :] = cumsum(rewards)
    end
    r_m_oracle = vec(mean(oracle_matrix, dims=1))

    p = plot(title="Regret Acumulado Médio (σ = $(round(sigma, digits=3)))",
             xlabel="Rodada (t)",
             ylabel="Regret R(t)",
             legend=:outerright,
             size=(850, 480),
             left_margin=8Plots.mm,
             bottom_margin=5Plots.mm)
    
    for (idx, (label_name, policy_fn)) in enumerate(policies)
        c = palette(:default)[idx]
        accumulated_matrix = zeros(Float64, R, T)
        
        for rep in 1:R
            rewards, _ = run_test(bandit, T, policy_fn)
            accumulated_matrix[rep, :] = cumsum(rewards)
        end
        
        r_m_policy = vec(mean(accumulated_matrix, dims=1))
        regret = r_m_oracle .- r_m_policy
        
        plot!(p, t, regret, label=label_name, color=c, lw=2)
    end
    
    display(p)
    sleep(1.0)
end

# --- 3. Porcentagem Instantânea da Ação Ótima ---
for sigma in sigmas
    bandit = [[mu, sigma] for mu in mus]
    
    p = plot(title="% Escolha da Ação Ótima (σ = $(round(sigma, digits=3)))",
             xlabel="Rodada (t)",
             ylabel="% Braço Ótimo",
             ylims=(0, 105),
             legend=:outerright,
             size=(850, 480),
             left_margin=8Plots.mm,
             bottom_margin=5Plots.mm)
    
    for (idx, (label_name, policy_fn)) in enumerate(policies_with_oracle)
        c = idx == 1 ? :black : palette(:default)[idx - 1]
        ls = idx == 1 ? :dash : :solid

        hit_matrix = zeros(Float64, R, T)
        for rep in 1:R
            _, is_optimal = run_test(bandit, T, policy_fn)
            hit_matrix[rep, :] = is_optimal
        end
        
        pct_optimal = vec(mean(hit_matrix, dims=1)) .* 100
        plot!(p, t, pct_optimal, label=label_name, color=c, linestyle=ls, lw=2)
    end
    
    display(p)
    sleep(1.0)
end