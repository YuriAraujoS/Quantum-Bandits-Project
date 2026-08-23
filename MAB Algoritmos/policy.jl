#nesse arquivo definimos cada forma de escolher os braços

function policy_epsilon_greedy(bandit, estimates, eps)
    if rand() < eps
        return rand(eachindex(bandit))
    else
        return argmax_rand([e[1] for e in estimates])
    end
end

function policy_epsilon_decay(bandit, estimates,eps, n, t)

    if  rand() < min(1.0, eps * n /t)
        return rand(eachindex(bandit))
    else
        return argmax_rand([e[1] for e in estimates])
    end

end