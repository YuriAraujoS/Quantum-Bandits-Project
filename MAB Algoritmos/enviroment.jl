#função que cria os bandits e os retorna
function create_bandit(number_arms)
    #definindo os mus e sigmas de cada bandit aleatoriamente
    mus = [1.0 + (5.0 - 1.0) * rand() for _ in 1:number_arms]
    sigmas = [0.5 + (3-0.5) * rand() for _ in 1:number_arms]
    #criando um array de bandits com seus mus e sigmas
    bandit = [[mus[i], sigmas[i]] for i in 1:number_arms]
    return bandit
end

#função que retorna a recompensa por girar determinado braço e atualiza suas estimativas
function action(arm, estimates,bandit)
    #cálculo matemático da recompensa do braço
    reward = bandit[arm][1] + bandit[arm][2] * randn()
    #atualização das estimativas de acordo com as anteriores e a atual
    estimates[arm][2] += 1
    estimates[arm][1] += (1 / estimates[arm][2]) * (reward - estimates[arm][1])
    return reward
end

#função que resolve desempates aleatóriamente
function argmax_rand(Q)
    max_val = maximum(Q)
    best_arms = findall(==(max_val), Q)
    return rand(best_arms)
end

#função que inicializa todos os métodos de estimativa

function run_test(bandit, n, policy_fn)
    #criando array de estimativas e numero de passos
    #o primeiro número representa as recompensas estimadas e o segundo o número de passos
    estimates = [[0.0,0] for _ in bandit]

    #criando o vetor de recompensas para plotar gráficos depois
    rewards = zeros(Float64, n)

    #verificando a melhor ação de cara para plotar gráficos depois
    best_action = argmax([b[1] for b in bandit])

    #vetor para calcular a percentagem acumulada de melhores ações
    is_optimal = zeros(Int, n)

    for i in 1:n
    
    #seleciona de acordo com a política passada na hora de escolher os parâmetros
    arm = policy_fn(bandit, estimates, i)

    if arm == best_action
        is_optimal[i] = 1
    end

    rewards[i] = action(arm, estimates, bandit)
    end
    return rewards, is_optimal

end