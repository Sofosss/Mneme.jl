include(joinpath(@__DIR__, "torcjulia", "torcjulia.jl"))
import .torcjulia

include(joinpath(@__DIR__, "src", "Mneme.jl"))
import .Mneme

using PythonCall

np = pyimport("numpy")

using CSV, Tables

function fetch_data(file::String, features::Vector{Symbol})
    tbl = CSV.File(file; select = features)
    Tables.matrix(tbl)
end


function main()

    path = "./data/data.csv"; num_blocks = 5
    reader = Mneme.BlockReader(path; num_blocks = num_blocks, target = [:Gender])

    data = fetch_data(path, [:Name, :Score, :Age, :Gender])

    std_scaler = Mneme.StandardScaler(path, [:Score, :Age])
    Mneme.fit(std_scaler, reader)
    Mneme.print_stats(std_scaler)

    minmax_scaler = Mneme.MinMaxScaler(path, [:Score])
    Mneme.fit(minmax_scaler, reader)
    Mneme.print_stats(minmax_scaler)

    maxabs_scaler = Mneme.MaxAbsScaler(path, [:Age])
    Mneme.fit(maxabs_scaler, reader)
    Mneme.print_stats(maxabs_scaler)

    std_scaler = Mneme.StandardScaler(path, [:Score, :Age]; with_std = true)
    Mneme.fit(std_scaler, reader)
    Mneme.print_stats(std_scaler)

    ordinal_encoder = Mneme.OrdinalEncoder(path, [:Name])
    Mneme.fit(ordinal_encoder, reader)

    Mneme.print_stats(ordinal_encoder)

    pipeline = Mneme.Pipeline([Mneme.MinMaxScaler(path, [:Age, :Score,]),
                              Mneme.OrdinalEncoder(path, [:Gender, :Name])], path)
    Mneme.fit(pipeline, reader)
    Mneme.print_stats(pipeline)
    
    print(Mneme.transform(pipeline, data))

end

torcjulia.start(main)