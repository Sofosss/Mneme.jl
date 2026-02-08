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

    # mm_data_np = Mneme.transform(minmax_scaler, data[:, 2:3])
    # mm_data = pyconvert(Array{Float64}, mm_data_np)
    # println(mm_data[1:10, :])

    maxabs_scaler = Mneme.MaxAbsScaler(path, [:Age])
    Mneme.fit(maxabs_scaler, reader)
    Mneme.print_stats(maxabs_scaler)

    # maxabs_data_np = Mneme.transform(maxabs_scaler, data[:, 2:3])
    # maxabs_data = pyconvert(Array{Float64}, maxabs_data_np)
    # println(maxabs_data[1:10, :])

    # std_scaler = Mneme.StandardScaler(path, [:Score, :Age]; with_std = true)
    # Mneme.fit(std_scaler, reader)
    # Mneme.print_stats(std_scaler)

    # std_data_np = Mneme.transform(std_scaler, data[:, 2:3])
    # std_data = pyconvert(Array{Float64}, std_data_np)
    # println(std_data[1:10, :])

    ordinal_encoder = Mneme.LabelEncoder(path, :Name)
    Mneme.fit(ordinal_encoder, reader)

    Mneme.print_stats(ordinal_encoder)

    # data[2, 1] = "Aaron"

    # ord_data_np = Mneme.transform(ordinal_encoder, data[:, 1:1])
    # ord_data = pyconvert(Array{Int}, ord_data_np)
    # println(ord_data[1:10, :])

    # label_encoder = Mneme.LabelEncoder(path, :Name)
    # Mneme.fit(label_encoder, reader)
    # Mneme.print_stats(label_encoder)

    # label_data_np = Mneme.transform(label_encoder, data[:, 1:1])
    # label_data = pyconvert(Array{Int}, label_data_np)
    # println(label_data[1:10, :])

    # sparse_output = true

    # # onehot_encoder = Mneme.OneHotEncoder(path, [:Gender]; sparse_output = sparse_output)
    # onehot_encoder = Mneme.OneHotEncoder(path, [:Gender]; sparse_output = sparse_output)
    # #onehot_encoder = Mneme.OneHotEncoder(path, [:Gender]; handle_unknown = "ignore", sparse_output = sparse_output)
    # #onehot_encoder = Mneme.OneHotEncoder(path, [:Gender]; handle_unknown = "ignore", sparse_output = sparse_output)
    # #onehot_encoder = Mneme.OneHotEncoder(path, [:Gender]; handle_unknown = "error",  sparse_output = sparse_output)
    # #onehot_encoder = Mneme.OneHotEncoder(path, [:Gender]; drop = nothing, sparse_output = sparse_output)
    # #onehot_encoder = Mneme.OneHotEncoder(path, [:Gender]; drop = ["Female"], sparse_output = sparse_output)
    # #onehot_encoder  = Mneme.OneHotEncoder(path, [:Gender]; drop = "if_binary", sparse_output = sparse_output)
    # #onehot_encoder = Mneme.OneHotEncoder(path, [:Gender]; drop = "first", handle_unknown = "ignore", sparse_output = sparse_output)
    # #onehot_encoder = Mneme.OneHotEncoder(path, [:Gender]; drop = "if_binary", handle_unknown = "ignore", sparse_output = sparse_output)
    # #onehot_encoder = Mneme.OneHotEncoder(path, [:Gender]; dtype = Float32, sparse_output = sparse_output)
    # #onehot_encoder = Mneme.OneHotEncoder(path, [:Gender]; dtype = Int32,   sparse_output = sparse_output)
    # #onehot_encoder = Mneme.OneHotEncoder(path, [:Gender, :Name]; handle_unknown = "ignore", sparse_output = sparse_output)
    # #onehot_encoder = Mneme.OneHotEncoder(path, [:Gender, :Name]; drop = "first", sparse_output = sparse_output)
    # #onehot_encoder = Mneme.OneHotEncoder(path, [:Gender]; categories = [["Female","Male"]], sparse_output = sparse_output)

    # Mneme.fit(onehot_encoder, reader)
    # Mneme.print_stats(onehot_encoder)

    # py_data = Mneme.transform(onehot_encoder, data[:, 4:4])
    
    # jl_data = sparse_output ? Mneme.to_JuliaCSR(py_data) : pyconvert(Array{Float64}, py_data)

    # if sparse_output
    #     println("indptr: ", jl_data.rowptr[1:10])
    #     println("indices: ", jl_data.colval[1:10])
    #     println("data: ", jl_data.nzval[1:10])
    # else
    #     println("data: ", jl_data[1:10, :])
    # end


    pipeline = Mneme.Pipeline([Mneme.StandardScaler(path, [:Score, :Age]; with_std = true), 
                              Mneme.MaxAbsScaler(path, [:Age]), Mneme.MinMaxScaler(path, [:Score]),
                              Mneme.OrdinalEncoder(path, [:Name])], path)
    Mneme.fit(pipeline, reader)
    Mneme.print_stats(pipeline)

end

torcjulia.start(main)