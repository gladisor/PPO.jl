θ = range(-2pi, 2pi, length = 100)
θ̇ = range(-2, 2, length = 100)

s = Matrix{Vector}(undef, (length(θ), length(θ̇)))
for i ∈ axes(s, 1)
    for j ∈ axes(s, 2)
        s[i, j] = [θ[i], θ̇[j]]
    end
end

f = x -> [cos(x[1]), sin(x[1]), x[2]]
s = f.(s)
value_img = ppo.critic.(s)
z = x -> x[1]
value_img = z.(value_img)
savefig(heatmap(value_img, xlabel = "Angle", ylabel = "Angular Velocity"), "heatmap")
