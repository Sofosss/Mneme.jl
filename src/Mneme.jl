module Mneme

import ..torcjulia

include("offsets.jl")

include("./preprocessors/scalers/minmaxscaler.jl")
include("./preprocessors/scalers/maxabsscaler.jl")
include("./preprocessors/scalers/standardscaler.jl")

include("./preprocessors/encoders/ordinalencoder.jl")
include("./preprocessors/encoders/labelencoder.jl")
include("./preprocessors/encoders/onehotencoder.jl")

include("./preprocessors/pipeline.jl")


export BlockReader, Pipeline
export MinMaxScaler, MaxAbsScaler, StandardScaler
export OrdinalEncoder, LabelEncoder, OneHotEncoder
export fit, transform, print_stats, to_JuliaCSR

using PythonCall

function init(func::Function)
    torcjulia.start(func)       
end

end