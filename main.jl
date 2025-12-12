include(joinpath(@__DIR__, "torcjulia", "torcjulia.jl"))
import .torcjulia

include(joinpath(@__DIR__, "src", "Mneme.jl"))
import .Mneme


function main()

    path = "./data/data.csv"; num_blocks = 5
    reader = Mneme.BlockReader(path; num_blocks = num_blocks, target = [:Gender])

    minmax_scaler = Mneme.MinMaxScaler(path, [:Score, :Age])
    Mneme.fit(minmax_scaler, reader)
end

torcjulia.start(main)
