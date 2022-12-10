using Plots
ENV["GKSwstype"] = "nul"

using ReinforcementLearning
using Flux
using Flux: Optimiser
using Distributions
using BSON
include("render.jl")

struct Value
    layers::Chain
end

Flux.@functor Value

function Value(s_size::Int, h_size::Int)
    layers = Chain(
        Dense(s_size, h_size, tanh),
        Dense(h_size, h_size, tanh),
        Dense(h_size, 1))

    return Value(layers)
end

function (value::Value)(s)
    return value.layers(s)
end

struct Actor
    pre::Chain
    μ::Dense
    logσ::Dense
    a_lim::Real
end

Flux.@functor Actor
Flux.trainable(actor::Actor) = (actor.pre, actor.μ, actor.logσ)

function Actor(s_size::Int, h_size::Int, a_size::Int; a_lim::Real = 1.0)
    pre = Chain(
        Dense(s_size, h_size, tanh),
        Dense(h_size, h_size, tanh)
        )

    μ = Dense(h_size, a_size)
    logσ = Dense(h_size, a_size)

    return Actor(pre, μ, logσ, a_lim)
end

function (actor::Actor)(s)
    x = actor.pre(s)
    μ = actor.μ(x)
    σ = exp.(clamp.(actor.logσ(x), -20, 10))
    ϵ = randn(Float32, size(σ))
    a = tanh.(μ .+ σ .* ϵ) * actor.a_lim
    return a, μ, σ
end

mutable struct PPO <: AbstractPolicy
    actor
    critic
    target_actor
    target_critic

    actor_opt
    critic_opt

    actor_loss::Float32
    critic_loss::Float32

    γ::Float32
    ρ::Float32
end

function PPO(;actor, critic, actor_opt, critic_opt, γ, ρ)
    return PPO(
        actor, critic, 
        deepcopy(actor), deepcopy(critic), 
        actor_opt, critic_opt, 
        0.0f0, 0.0f0, γ, ρ)
end

function (ppo::PPO)(env::PendulumEnv)
    a, μ, σ = ppo.actor(state(env))
    return a[1]
end

function build_buffer(env::PendulumEnv, episodes::Int)

    episode_length = env.params.max_steps

    traj = CircularArraySARTTrajectory(
            capacity = episodes * episode_length,
            state = typeof(state(env)) => size(state(env)),
            action = Float32 => 1)

    return traj
end

function soft_update!(target, source, ρ)
    for (targ, src) in zip(Flux.params(target), Flux.params(source))
        targ .= ρ .* targ .+ (1 - ρ) .* src
    end
end

function sarts(traj::CircularArraySARTTrajectory)
    idx = 1:length(traj)
    s, a, r, t = (convert(Array, select_last_dim(traj[x], idx)) for x in SART)
    s′ = convert(Array, select_last_dim(traj[:state], idx .+ 1))

    return (s, a, r, t, s′)
end


path = "ppo_entropy"

ppo = PPO(
    actor = Actor(3, 64, 1, a_lim = 2.0f0),
    critic = Value(3, 128),
    actor_opt = Optimiser(ClipNorm(1f-4), Adam(0.00005f0)),
    critic_opt = Optimiser(ClipNorm(1f-4), Adam(0.0003f0)),
    γ = 0.95f0,
    ρ = 0.995f0)

γ = ppo.γ
env = PendulumEnv()
episodes = 20
epochs = 5
actor_ps = Flux.params(ppo.actor)
critic_ps = Flux.params(ppo.critic)

hook = TotalRewardPerEpisode()
actor_loss = []
critic_loss = []

for iteration ∈ 1:100

    agent = Agent(policy = ppo, trajectory = build_buffer(env, episodes))
    run(agent, env, StopAfterEpisode(episodes), hook)

    data = sarts(agent.trajectory)
    batches = Flux.DataLoader(data, batchsize = 32, shuffle = true)

    for epoch ∈ 1:epochs

        for batch ∈ batches

            s, a, r, t, s′ = batch

            critic_gs = Flux.gradient(critic_ps) do 
                y = r .+ γ * vec(ppo.target_critic(s′))
                δ = y .- vec(ppo.critic(s))
                loss = mean(δ .^ 2)

                Flux.ignore() do 
                    ppo.critic_loss = loss
                end
                return loss
            end

            Flux.update!(ppo.critic_opt, critic_ps, critic_gs)

            actor_gs = Flux.gradient(actor_ps) do
                _, μ_old, σ_old = ppo.target_actor(s)
                old_p_a = prod(pdf.(Normal.(μ_old, σ_old), a), dims = 1) |> vec

                _, μ, σ = ppo.actor(s)
                p_a = prod(pdf.(Normal.(μ, σ), a), dims = 1) |> vec
                ratio = p_a ./ old_p_a

                δ = r .+ γ * vec(ppo.target_critic(s′)) .- vec(ppo.critic(s))

                δ = (δ .- mean(δ)) ./ std(δ)

                loss = -mean(min.(ratio .* δ, clamp.(ratio, 0.9f0, 1.1f0) .* δ)) + mean((1 .- ratio) .^ 2) + mean(log.(p_a) ./ old_p_a) * 0.0001

                Flux.ignore() do 
                    ppo.actor_loss = loss
                end

                return loss
            end

            Flux.update!(ppo.actor_opt, actor_ps, actor_gs)

            soft_update!(ppo.target_critic, ppo.critic, ppo.ρ)
            push!(actor_loss, ppo.actor_loss)
            push!(critic_loss, ppo.critic_loss)
        end
    end

    soft_update!(ppo.target_actor, ppo.actor, 0.0f0)

    savefig(plot(actor_loss, label = false), joinpath(path, "actor_loss"))
    savefig(plot(critic_loss, label = false), joinpath(path, "critic_loss"))

    if iteration % 10 == 0
        run(ppo, env, StopWhenDone(), Render(path = joinpath(path, "animations/vid_$iteration.mp4")))
        BSON.@save joinpath(path, "models/ppo_$iteration.bson") ppo
    end
end

