workspace "HelloGPUCompute"
    location "build"
    configurations { "Debug", "Release" }

architecture "x86_64"

-- project "cudaEnv"
--     kind "ConsoleApp"
--     language "C++"
--     targetdir "bin/"
--     systemversion "latest"
--     flags { "MultiProcessorCompile", "NoPCH" }

--     -- Src
--     files { "cudaEnv.cu", "tinyhipradixsort.hpp" }

--     -- CUDA
--     buildcustomizations "BuildCustomizations/CUDA 12.0"
--     includedirs { "$(CUDA_PATH)/include" }
--     libdirs { "$(CUDA_PATH)/lib/x64" }
--     links { "cuda" }

--     symbols "On"

--     filter {"Debug"}
--         runtime "Debug"
--         targetname ("cudaEnv_Debug")
--         optimize "Off"
--     filter {"Release"}
--         runtime "Release"
--         targetname ("cudaEnv")
--         optimize "Full"
--     filter{}

project "unittest"
    kind "ConsoleApp"
    language "C++"
    targetdir "bin/"
    systemversion "latest"
    flags { "MultiProcessorCompile", "NoPCH" }

    -- Src
    files { "unittest.cpp",  "tinyhipradixsort.hpp" }

    -- Orochi
    includedirs { "libs/orochi" }
    files { "libs/orochi/Orochi/Orochi.h" }
    files { "libs/orochi/Orochi/Orochi.cpp" }
    includedirs { "libs/orochi/contrib/hipew/include" }
    files { "libs/orochi/contrib/hipew/src/hipew.cpp" }
    includedirs { "libs/orochi/contrib/cuew/include" }
    files { "libs/orochi/contrib/cuew/src/cuew.cpp" }
    links { "version" }

    -- Radix sort module 
    includedirs { "libs/orochi/ParallelPrimitives" }
    files { "libs/orochi/ParallelPrimitives/RadixSort.cpp" }
    files { "libs/orochi/Orochi/OrochiUtils.cpp" }
    defines{"__WINDOWS__", "NOMINMAX"}
    characterset ("ASCII")
    links { "kernel32" }


    -- UTF8
    postbuildcommands { 
        "{COPYFILE} ../libs/orochi/contrib/bin/win64/*.dll ../bin"
    }

    symbols "On"

    filter {"Debug"}
        runtime "Debug"
        targetname ("unittest_Debug")
        optimize "Off"
    filter {"Release"}
        runtime "Release"
        targetname ("unittest")
        optimize "Full"
    filter{}


project "main"
    kind "ConsoleApp"
    language "C++"
    targetdir "bin/"
    systemversion "latest"
    flags { "MultiProcessorCompile", "NoPCH" }

    -- Src
    files { "main.cpp", "kernel.cu", "tinyhipradixsort.hpp" }

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

project "helloworld"
    kind "ConsoleApp"
    language "C++"
    targetdir "bin/"
    systemversion "latest"
    flags { "MultiProcessorCompile", "NoPCH" }

    -- Src
    files { "helloworld.cpp", "tinyhipradixsort.hpp" }

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
        targetname ("helloworld_Debug")
        optimize "Off"
    filter {"Release"}
        runtime "Release"
        targetname ("helloworld")
        optimize "Full"
    filter{}