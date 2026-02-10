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

    OFFSETS_FILEPATH = "/path/to/offsets/file (.dat)"
    DATASET_FILEPATH = "/path/to/dataset/file (.csv)"
    NROWS = 100 # total number of dataset's samples
    NUM_BLOCKS = 5

    reader = Mneme.BlockReader(DATASET_FILEPATH; num_blocks = NUM_BLOCKS, target = [:Gender], 
                                                 num_rows = NROWS, offsets_path = OFFSETS_FILEPATH)

    data = fetch_data(DATASET_FILEPATH, [:Name, :Score, :Age, :Gender])

    std_scaler = Mneme.StandardScaler(DATASET_FILEPATH , [:Score, :Age])
    Mneme.fit(std_scaler, reader)
    Mneme.print_stats(std_scaler)

    minmax_scaler = Mneme.MinMaxScaler(DATASET_FILEPATH , [:Score])
    Mneme.fit(minmax_scaler, reader)
    Mneme.print_stats(minmax_scaler)

    maxabs_scaler = Mneme.MaxAbsScaler(DATASET_FILEPATH , [:Age])
    Mneme.fit(maxabs_scaler, reader)
    Mneme.print_stats(maxabs_scaler)

    std_scaler = Mneme.StandardScaler(DATASET_FILEPATH , [:Score, :Age]; with_std = true)
    Mneme.fit(std_scaler, reader)
    Mneme.print_stats(std_scaler)

    ordinal_encoder = Mneme.OrdinalEncoder(DATASET_FILEPATH , [:Name])
    Mneme.fit(ordinal_encoder, reader)
    Mneme.print_stats(ordinal_encoder)

    pipeline = Mneme.Pipeline([Mneme.MinMaxScaler(DATASET_FILEPATH , [:Age, :Score,]),
                              Mneme.OrdinalEncoder(DATASET_FILEPATH , [:Gender, :Name])], DATASET_FILEPATH)
    Mneme.fit(pipeline, reader)
    Mneme.print_stats(pipeline)
    
    print(Mneme.transform(pipeline, data))

end

torcjulia.start(main)