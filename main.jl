include(joinpath(@__DIR__, "torcjulia", "torcjulia.jl"))
import .torcjulia

include(joinpath(@__DIR__, "src", "Mneme.jl"))
import .Mneme

using PythonCall

np = pyimport("numpy")

using CSV, Tables

function fetch_data(file::String, features::Vector{Symbol})
    tbl = CSV.File(file; select = features, types = Float64)
    Tables.matrix(tbl)
end


function main()

    path = "./data/data.csv"; num_blocks = 5
    reader = Mneme.BlockReader(path; num_blocks = num_blocks, target = [:Gender])

    data = fetch_data(path, [:Score, :Age])

    minmax_scaler = Mneme.MinMaxScaler(path, [:Score, :Age])
    Mneme.fit(minmax_scaler, reader)
    Mneme.print_stats(minmax_scaler)

    mm_data_np = Mneme.transform(minmax_scaler, data)
    mm_data = pyconvert(Array{Float64}, mm_data_np)
    println(mm_data[1:10, :])

    maxabs_scaler = Mneme.MaxAbsScaler(path, [:Score, :Age])
    Mneme.fit(maxabs_scaler, reader)
    Mneme.print_stats(maxabs_scaler)

    maxabs_data_np = Mneme.transform(maxabs_scaler, data)
    maxabs_data = pyconvert(Array{Float64}, maxabs_data_np)
    println(maxabs_data[1:10, :])


end

torcjulia.start(main)
