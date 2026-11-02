# pgr4_recompiled - ReXGlue Recompiled Project
#
# SDK boilerplate lives in generated/rexglue.cmake.

cmake_minimum_required(VERSION 3.25)
project(pgr4_recompiled LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Renderer selection:
#   PGR4_D3D12_RENDERER=ON   Plume/D3D12 native renderer in src/gpu (hooks the
#                            guest D3D calls; no Xenos/PM4 emulation).
#   PGR4_NATIVE_RENDERER=ON  Vulkan native renderer (needs PGR4_NATIVE_RENDERER_ROOT).
#   both OFF                 SDK Xenos GPU plugin (rexgpu-xenos).
option(PGR4_D3D12_RENDERER "Build the Plume-backed native D3D12 renderer" OFF)
option(PGR4_NATIVE_RENDERER "Build with the PGR4 Vulkan native renderer instead of Xenos" ON)
set(PGR4_PLUME_DIR "C:/Users/Bea/Documents/GH-II-Recomp/thirdparty/plume" CACHE PATH "Plume source tree")

include(generated/rexglue.cmake)

# Sources
set(PGR4RECOMPILED_SOURCES
    src/main.cpp
    src/music_volume_hook.cpp
    src/ground_query_guard.cpp
    src/session_debug.cpp
    src/stream_flush_guard.cpp
    src/audio_lock_trace.cpp
    src/occlusion_query.cpp
    src/walk_camera_pace.cpp
    src/content/dlc.cpp
    src/profiler.cpp
)

if(PGR4_D3D12_RENDERER)
    list(APPEND PGR4RECOMPILED_SOURCES
        src/gpu/video.cpp
        src/gpu/imgui_drawer.cpp
        src/gpu/renderer.cpp
        src/gpu/shader_cache.cpp
        src/gpu/texture_convert.cpp
        src/gpu/hooks/device.cpp
        src/gpu/hooks/draw.cpp
        src/gpu/hooks/shader.cpp
        src/gpu/hooks/state.cpp
        src/gpu/hooks/gw.cpp
        src/gpu/fault_trace.cpp
    )
else()
    list(APPEND PGR4RECOMPILED_SOURCES src/xenos_debug.cpp)
endif()

if(WIN32)
    add_executable(pgr4_recompiled WIN32 ${PGR4RECOMPILED_SOURCES})
else()
    add_executable(pgr4_recompiled ${PGR4RECOMPILED_SOURCES})
endif()

target_include_directories(pgr4_recompiled PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src)

if(PGR4_D3D12_RENDERER)
    if(NOT WIN32)
        message(FATAL_ERROR "PGR4_D3D12_RENDERER targets D3D12 (Windows) only")
    endif()
    if(NOT EXISTS "${PGR4_PLUME_DIR}/CMakeLists.txt")
        message(FATAL_ERROR "Plume not found at ${PGR4_PLUME_DIR}; set PGR4_PLUME_DIR")
    endif()
    add_subdirectory("${PGR4_PLUME_DIR}" plume EXCLUDE_FROM_ALL)

    # Debug info for the renderer and Plume only (the generated guest code is
    # left alone), so render thread samples (profile_render_thread) resolve.
    option(PGR4_PROFILE_SYMBOLS "Debug info for the renderer and Plume, for CPU profiling" OFF)
    if(PGR4_PROFILE_SYMBOLS)
        set_source_files_properties(src/gpu/renderer.cpp src/gpu/video.cpp src/gpu/texture_convert.cpp
            src/gpu/shader_cache.cpp PROPERTIES COMPILE_OPTIONS "-g;-gcodeview")
        target_compile_options(plume PRIVATE -g -gcodeview)
        target_link_options(pgr4_recompiled PRIVATE "LINKER:/DEBUG")
    endif()

    # Host shaders (ImGui overlay) compiled to shader model 5.1 DXBC headers
    # with fxc, which every D3D12 GPU accepts (DXIL needs SM 6.0 drivers).
    find_program(PGR4_FXC fxc
        HINTS "$ENV{ProgramFiles\(x86\)}/Windows Kits/10/bin/10.0.26100.0/x64"
        REQUIRED)
    set(PGR4_HOST_SHADER_DIR "${CMAKE_CURRENT_BINARY_DIR}/host_shaders")
    set(PGR4_HOST_SHADER_HEADERS)
    foreach(shader IN ITEMS imgui_vs:vs_5_1 imgui_ps:ps_5_1 present_vs:vs_5_1 present_ps:ps_5_1
                        solid_ps:ps_5_1)
        string(REPLACE ":" ";" parts "${shader}")
        list(GET parts 0 stem)
        list(GET parts 1 profile)
        set(out "${PGR4_HOST_SHADER_DIR}/${stem}.dxbc.h")
        add_custom_command(
            OUTPUT "${out}"
            COMMAND ${CMAKE_COMMAND} -E make_directory "${PGR4_HOST_SHADER_DIR}"
            COMMAND "${PGR4_FXC}" /nologo /T ${profile} /E main /O3 /Fh "${out}" /Vn g_${stem}_dxbc
                    "${CMAKE_CURRENT_SOURCE_DIR}/src/gpu/shaders/${stem}.hlsl"
            DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/src/gpu/shaders/${stem}.hlsl"
            COMMENT "fxc ${stem}.hlsl (${profile})"
            VERBATIM)
        list(APPEND PGR4_HOST_SHADER_HEADERS "${out}")
    endforeach()
    target_sources(pgr4_recompiled PRIVATE ${PGR4_HOST_SHADER_HEADERS})
    target_include_directories(pgr4_recompiled PRIVATE "${PGR4_HOST_SHADER_DIR}")

    # Guest shader cache: the XenosRecomp HLSL in shaders/hlsl (see
    # tools/shaders/README.md) compiled to shader model 5.1 DXBC and packed
    # next to the executable.
    find_package(Python3 REQUIRED COMPONENTS Interpreter)
    file(GLOB PGR4_GUEST_HLSL CONFIGURE_DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/shaders/hlsl/*.hlsl")
    set(PGR4_SHADER_CACHE "${CMAKE_CURRENT_BINARY_DIR}/shader_cache_sm51.bin")
    add_custom_command(
        OUTPUT "${PGR4_SHADER_CACHE}"
        COMMAND Python3::Interpreter "${CMAKE_CURRENT_SOURCE_DIR}/tools/shaders/build_shader_cache.py"
                --fxc "${PGR4_FXC}" --work "${CMAKE_CURRENT_BINARY_DIR}/shader_cache_work"
                --out "${PGR4_SHADER_CACHE}"
        DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/tools/shaders/build_shader_cache.py"
                "${CMAKE_CURRENT_SOURCE_DIR}/tools/shaders/shader_common_sm51.h"
                ${PGR4_GUEST_HLSL}
        COMMENT "Building the shader model 5.1 guest shader cache"
        VERBATIM)
    # Same HLSL compiled to DXIL (shader model 6.0) for GPUs that support it;
    # the renderer picks it at startup (ShaderCache::Load).
    set(PGR4_SHADER_CACHE_DXIL "${CMAKE_CURRENT_BINARY_DIR}/shader_cache_dxil.bin")
    add_custom_command(
        OUTPUT "${PGR4_SHADER_CACHE_DXIL}"
        COMMAND Python3::Interpreter "${CMAKE_CURRENT_SOURCE_DIR}/tools/shaders/build_shader_cache.py"
                --dxil --work "${CMAKE_CURRENT_BINARY_DIR}/shader_cache_work_dxil"
                --out "${PGR4_SHADER_CACHE_DXIL}"
        DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/tools/shaders/build_shader_cache.py"
                "${CMAKE_CURRENT_SOURCE_DIR}/tools/shaders/shader_common_sm51.h"
                ${PGR4_GUEST_HLSL}
        COMMENT "Building the shader model 6.0 (DXIL) guest shader cache"
        VERBATIM)
    add_custom_target(pgr4_shader_cache DEPENDS "${PGR4_SHADER_CACHE}" "${PGR4_SHADER_CACHE_DXIL}")
    add_dependencies(pgr4_recompiled pgr4_shader_cache)
    add_custom_command(TARGET pgr4_recompiled POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different "${PGR4_SHADER_CACHE}" "${PGR4_SHADER_CACHE_DXIL}"
                "$<TARGET_FILE_DIR:pgr4_recompiled>"
        VERBATIM)
    rexglue_setup_target(pgr4_recompiled)
    target_link_libraries(pgr4_recompiled PRIVATE plume d3d12 dxgi dxguid)
    target_compile_definitions(pgr4_recompiled PRIVATE PGR4_D3D12_RENDERER=1 PGR4_NATIVE_RENDERER=0)
elseif(PGR4_NATIVE_RENDERER)
    rexglue_setup_target(pgr4_recompiled)
    target_compile_definitions(pgr4_recompiled PRIVATE PGR4_D3D12_RENDERER=0 PGR4_NATIVE_RENDERER=1)
    include("${PGR4_NATIVE_RENDERER_ROOT}/native_renderer/integration/pgr4_native_renderer.cmake")
else()
    rexglue_setup_target(pgr4_recompiled GPU_PLUGINS xenos)
    target_compile_definitions(pgr4_recompiled PRIVATE PGR4_D3D12_RENDERER=0 PGR4_NATIVE_RENDERER=0)
endif()

# Geometry Wars: Waves (gw.xex), the arcade machine in PGR4's garage. PGR4
# launches it with XamLoaderLaunchTitle; the app switches to it in-process
# (src/pgr4_recompiled_app.h). Its recompiled code is a second title in this
# exe, generated from gw/gw_manifest.toml with every symbol prefixed gw_, and
# built with its own PCH (its image configuration macros differ from PGR4's).
# Its codegen lives in gw/ so its glue never replaces generated/rexglue.cmake.
if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/gw/generated/sources.cmake")
    include(gw/generated/sources.cmake)
    set(PGR4_GW_SOURCES ${GENERATED_SOURCES})
    set_source_files_properties(${PGR4_GW_SOURCES} PROPERTIES COMPILE_OPTIONS "${REXGLUE_RECOMP_OPTIONS}")
    add_custom_command(
        OUTPUT "${CMAKE_CURRENT_SOURCE_DIR}/gw/generated/codegen.build.stamp" ${PGR4_GW_SOURCES}
        COMMAND $<TARGET_FILE:rex::rexglue> codegen ${CMAKE_CURRENT_SOURCE_DIR}/gw/gw_manifest.toml
        DEPFILE "${CMAKE_CURRENT_SOURCE_DIR}/gw/generated/codegen.d"
        WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/gw
        COMMENT "Generating recompiled code for Geometry Wars"
        VERBATIM)
    add_custom_target(pgr4_gw_codegen DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/gw/generated/codegen.build.stamp")
    add_library(pgr4_gw_recomp OBJECT ${PGR4_GW_SOURCES})
    target_include_directories(pgr4_gw_recomp PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/gw/generated)
    target_link_libraries(pgr4_gw_recomp PRIVATE rex::runtime)
    rexglue_apply_target_settings(pgr4_gw_recomp)
    rexglue_apply_recomp_settings(pgr4_gw_recomp "${CMAKE_CURRENT_SOURCE_DIR}/gw/generated/gw_pch.h")
    add_dependencies(pgr4_gw_recomp pgr4_gw_codegen)
    target_link_libraries(pgr4_recompiled PRIVATE pgr4_gw_recomp)
endif()

# Online System Link over Epic Online Services (any renderer, Xenos too): the
# runtime loads this DLL and reads eos_credentials.ini beside the exe; both
# are optional and only used with --net_transport=eos.
set(PGR4_EOS_SDK_DIR "C:/Users/Bea/Documents/EOS-SDK/SDK" CACHE PATH "EOS SDK folder")
if(EXISTS "${PGR4_EOS_SDK_DIR}/Bin/EOSSDK-Win64-Shipping.dll")
    add_custom_command(TARGET pgr4_recompiled POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different "${PGR4_EOS_SDK_DIR}/Bin/EOSSDK-Win64-Shipping.dll"
                "$<TARGET_FILE_DIR:pgr4_recompiled>"
        VERBATIM)
endif()
if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/eos_credentials.ini")
    add_custom_command(TARGET pgr4_recompiled POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different "${CMAKE_CURRENT_SOURCE_DIR}/eos_credentials.ini"
                "$<TARGET_FILE_DIR:pgr4_recompiled>"
        VERBATIM)
endif()

# Launcher: game folder, GPU / shader format, title update and DLC packages.
# Static runtime and no extra DLLs, so it runs as is under Proton.
if(WIN32)
    add_executable(pgr4_launcher WIN32 tools/launcher/launcher.cpp tools/launcher/launcher.rc)
    target_compile_features(pgr4_launcher PRIVATE cxx_std_20)
    target_compile_definitions(pgr4_launcher PRIVATE UNICODE _UNICODE)
    set_property(TARGET pgr4_launcher PROPERTY MSVC_RUNTIME_LIBRARY "MultiThreaded")
    # Hybrid CRT: vcruntime linked in, the UCRT from the system (Windows 10+
    # and Wine both ship it). Own map file: the global /MAP names the game's.
    target_link_options(pgr4_launcher PRIVATE "LINKER:/MANIFEST:EMBED" "LINKER:/NODEFAULTLIB:libucrt.lib"
                        "LINKER:/MAP:pgr4_launcher.map")
endif()
