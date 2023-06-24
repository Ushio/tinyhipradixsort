workspace "HelloGPUCompute"
    location "build"
    configurations { "Debug", "Release" }

architecture "x86_64"

project "cudaEnv"
    kind "ConsoleApp"
    language "C++"
    targetdir "bin/"
    systemversion "latest"
    flags { "MultiProcessorCompile", "NoPCH" }

    -- Src
    files { "cudaEnv.cu", "tinyhipradixsort.hpp" }

    -- CUDA
    buildcustomizations "BuildCustomizations/CUDA 12.0"
    includedirs { "$(CUDA_PATH)/include" }
    libdirs { "$(CUDA_PATH)/lib/x64" }
    links { "cuda" }

    -- filter {"Debug"}
    --     links { "prlib_d" }
    -- UTF8
    -- postbuildcommands { 
    --     "{COPYFILE} ../libs/orochi/contrib/bin/win64/*.dll ../bin"
    -- }

    symbols "On"

    filter {"Debug"}
        runtime "Debug"
        targetname ("cudaEnv_Debug")
        optimize "Off"
    filter {"Release"}
        runtime "Release"
        targetname ("cudaEnv")
        optimize "Full"
    filter{}

project "main"
    kind "ConsoleApp"
    language "C++"
    targetdir "bin/"
    systemversion "latest"
    flags { "MultiProcessorCompile", "NoPCH" }

    -- Src
    files { "main.cpp", "shader.hpp", "kernel.cu", "tinyhipradixsort.hpp" }

    -- Orochi
    includedirs { "libs/orochi" }
    files { "libs/orochi/Orochi/Orochi.h" }
    files { "libs/orochi/Orochi/Orochi.cpp" }
    includedirs { "libs/orochi/contrib/hipew/include" }
    files { "libs/orochi/contrib/hipew/src/hipew.cpp" }
    includedirs { "libs/orochi/contrib/cuew/include" }
    files { "libs/orochi/contrib/cuew/src/cuew.cpp" }
    links { "version" }

    -- UTF8
    postbuildcommands { 
        "{COPYFILE} ../libs/orochi/contrib/bin/win64/*.dll ../bin"
    }

    symbols "On"

    filter {"Debug"}
        runtime "Debug"
        targetname ("main_Debug")
        optimize "Off"
    filter {"Release"}
        runtime "Release"
        targetname ("main")
        optimize "Full"
    filter{}