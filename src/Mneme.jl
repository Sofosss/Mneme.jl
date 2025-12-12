module Mneme

# include(joinpath(@__DIR__, "..", "torcjulia", "torcjulia.jl"))
import ..torcjulia

include("offsets.jl")
import .offsets: BlockReader, get_offsets, get_num_blocks, save_offsets


include("./preprocessors/minmaxscaler.jl")
import .minmaxscaler: MinMaxScaler, fit


function init(func::Function)
    torcjulia.start(func)       
end

end