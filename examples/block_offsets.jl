include(joinpath(@__DIR__, "..", "src", "Mneme.jl"))
import .Mneme

using DataFrames, CSV

path = "../data/data.csv"; num_blocks = 5
reader = Mneme.BlockReader(path; num_blocks = num_blocks, target = [:Gender])

offsets = Mneme.get_offsets(reader)

println(offsets)

Mneme.save_offsets(offsets; path = "../offsets/test_offsets")

reader = Mneme.BlockReader(path; num_blocks = num_blocks, target = [:Gender], offsets_path = "../offsets/test_offsets")

offsets = Mneme.get_offsets(reader)

println(offsets)

cols = CSV.read(path, DataFrame; limit = 0) |> names
blocks = Vector{DataFrame}(undef, Mneme.get_num_blocks(reader))

open(path, "r") do f
    for (i, offset) in enumerate(offsets)
        seek(f, offset)
        rows = String[]
        for j in 1:reader.block_size
            eof(f) && break
            push!(rows, readline(f))
        end

        blocks[i] = CSV.read(IOBuffer(join(rows, "\n")), DataFrame; header = cols)
    end
end

for (i, block) in enumerate(blocks)
    println("Block $i:")
    println(block)
    println("------------")
end