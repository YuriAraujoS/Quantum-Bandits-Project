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

function policy_ucb(bandit, estimates, t, c=2.0)
    for i in eachindex(bandit)
        if estimates[i][2] == 0
            return i
        end
    end
    ucb_values = [
        estimates[i][1] + c * sqrt(log(t) / estimates[i][2])
        for i in eachindex(bandit)
    ]

    return argmax_rand(ucb_values)
end

function policy_oracle(bandit, estimates, t)
    return argmax_rand([b[1] for b in bandit])
end