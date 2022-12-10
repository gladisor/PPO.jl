Base.@kwdef mutable struct Render <: AbstractHook
    a::Union{Animation, Nothing} = nothing
    path::String = "vid.mp4"
end

function (r::Render)(::PreEpisodeStage, agent, env)
    r.a = Animation()
    frame(r.a, plot(env))
end

function (r::Render)(::PostActStage, agent, env)
    frame(r.a, plot(env))
end

function (r::Render)(::PostEpisodeStage, agent, env)
    gif(r.a, r.path, fps = 20)
end