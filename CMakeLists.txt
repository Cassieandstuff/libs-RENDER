# ── RENDER ────────────────────────────────────────────────────────────────────
# C++23: a backend-neutral scene renderer interface and its Diligent Engine backend (Vulkan / Direct3D 12).
#
# Consumed via FetchContent; two targets:
#   render::api       — <render/Render.h>: ISceneRenderer, IOverlayRenderer, mesh / surface descriptors, typed handles,
#                       GpuAssetCache. Header-only, glm only: asset code builds against it with no graphics API.
#   render::diligent  — <render/Diligent.h>: RenderDevice, ViewportSurface, OffscreenTarget, DiligentSceneRenderer.
#                       Diligent stays PRIVATE (no Diligent type crosses the header).
# Layout:
#   API/render/  — public headers
#   src/         — the Diligent backend
#   tests/       — CTest suite (GPU tests carry the label `gpu`; `ctest -LE gpu` skips them)
#
# Needs glm (find_package(glm CONFIG) from vcpkg, or a `glm::glm` target the consumer defines). Diligent comes from the
# consumer when it already has a DiligentFX target, else it is fetched at the pins below; Diligent's configure step runs
# Python 3. The MSVC runtime is NOT set here: it inherits the consumer's.

cmake_minimum_required(VERSION 3.28)

project(RENDER
    VERSION 0.1.0
    DESCRIPTION "Backend-neutral scene renderer interface and its Diligent Engine backend"
    LANGUAGES C CXX)

if(PROJECT_IS_TOP_LEVEL)
    set(CMAKE_CXX_STANDARD 23)
    set(CMAKE_CXX_STANDARD_REQUIRED ON)
    set(CMAKE_CXX_EXTENSIONS OFF)
endif()

option(RENDER_DILIGENT    "Build the Diligent backend (render::diligent)" ON)
option(RENDER_BUILD_TESTS "Build RENDER's tests" ${PROJECT_IS_TOP_LEVEL})

if(NOT TARGET glm::glm)
    find_package(glm CONFIG REQUIRED)
endif()

# ── render::api ──────────────────────────────────────────────────────────────
add_library(render-api INTERFACE)
add_library(render::api ALIAS render-api)
target_include_directories(render-api INTERFACE $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/API>)
target_link_libraries(render-api INTERFACE glm::glm)
target_compile_features(render-api INTERFACE cxx_std_23)

if(NOT RENDER_DILIGENT)
    return()
endif()

# ── Diligent ─────────────────────────────────────────────────────────────────
if(NOT TARGET DiligentFX)
    # Diligent's configure step pip-installs libclang + jinja2 into whatever Python it finds; a build-local venv keeps
    # that out of the system Python.
    set(_venv "${CMAKE_BINARY_DIR}/_python-venv")
    if(NOT EXISTS "${_venv}/Scripts/python.exe" AND NOT EXISTS "${_venv}/bin/python")
        find_package(Python3 3.8 REQUIRED COMPONENTS Interpreter)
        execute_process(COMMAND "${Python3_EXECUTABLE}" -m venv "${_venv}")
        unset(Python3_EXECUTABLE CACHE)
    endif()
    unset(_Python3_EXECUTABLE CACHE)
    if(WIN32)
        set(Python3_EXECUTABLE "${_venv}/Scripts/python.exe" CACHE FILEPATH "" FORCE)
    else()
        set(Python3_EXECUTABLE "${_venv}/bin/python" CACHE FILEPATH "" FORCE)
    endif()

    set(DILIGENT_NO_DIRECT3D11            ON  CACHE BOOL "" FORCE)
    set(DILIGENT_NO_OPENGL                ON  CACHE BOOL "" FORCE)
    set(DILIGENT_NO_METAL                 ON  CACHE BOOL "" FORCE)
    set(DILIGENT_NO_ARCHIVER              ON  CACHE BOOL "" FORCE)
    set(DILIGENT_NO_RENDER_STATE_PACKAGER ON  CACHE BOOL "" FORCE)
    set(DILIGENT_NO_RADIENT               ON  CACHE BOOL "" FORCE)
    set(DILIGENT_BUILD_TESTS              OFF CACHE BOOL "" FORCE)
    set(DILIGENT_INSTALL_CORE             OFF CACHE BOOL "" FORCE)
    set(DILIGENT_INSTALL_TOOLS            OFF CACHE BOOL "" FORCE)
    set(DILIGENT_INSTALL_FX               OFF CACHE BOOL "" FORCE)
    include(FetchContent)
    # Side by side under DiligentEngine/ (the super-repo layout): Tools' and FX's headers reach Core as
    # "../../../DiligentCore/…".
    set(_droot "${FETCHCONTENT_BASE_DIR}/DiligentEngine")
    FetchContent_Declare(DiligentCore
        GIT_REPOSITORY https://github.com/DiligentGraphics/DiligentCore.git
        GIT_TAG        7f68fcfd9fb12ada6b4fa832244e0f6c14a6d190
        SOURCE_DIR     "${_droot}/DiligentCore"
        EXCLUDE_FROM_ALL)
    FetchContent_Declare(DiligentTools
        GIT_REPOSITORY https://github.com/DiligentGraphics/DiligentTools.git
        GIT_TAG        c84a6ff62789f7a167317448da6c62e923569588
        SOURCE_DIR     "${_droot}/DiligentTools"
        EXCLUDE_FROM_ALL)
    FetchContent_Declare(DiligentFX
        GIT_REPOSITORY https://github.com/DiligentGraphics/DiligentFX.git
        GIT_TAG        0984f522f55720997e8d12e4618130b07955d8fa
        SOURCE_DIR     "${_droot}/DiligentFX"
        EXCLUDE_FROM_ALL)
    FetchContent_MakeAvailable(DiligentCore DiligentTools DiligentFX)
endif()

# ── render::diligent ─────────────────────────────────────────────────────────
file(GLOB RENDER_DILIGENT_SOURCES CONFIGURE_DEPENDS
    "${CMAKE_CURRENT_SOURCE_DIR}/src/*.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/*.h")
add_library(render-diligent STATIC ${RENDER_DILIGENT_SOURCES} "${CMAKE_CURRENT_SOURCE_DIR}/API/render/Diligent.h")
add_library(render::diligent ALIAS render-diligent)
set_target_properties(render-diligent PROPERTIES CXX_STANDARD 23 CXX_STANDARD_REQUIRED ON CXX_EXTENSIONS OFF)
target_include_directories(render-diligent PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/src")
target_link_libraries(render-diligent
    PUBLIC  render::api glm::glm
    PRIVATE Diligent-GraphicsEngineVk-static
            Diligent-GraphicsEngineD3D12-static
            Diligent-GraphicsTools
            Diligent-TextureLoader
            Diligent-Common
            DiligentFX)
if(MSVC)
    target_compile_options(render-diligent PRIVATE /W4 /wd4100 /wd4505 /MP /utf-8)
endif()
source_group(TREE "${CMAKE_CURRENT_SOURCE_DIR}" FILES ${RENDER_DILIGENT_SOURCES})

# ── Tests ────────────────────────────────────────────────────────────────────
# The GPU tests render through each backend into an off-screen target and check the read-back pixels; they need a GPU
# and drivers, hence the `gpu` label.
if(RENDER_BUILD_TESTS)
    enable_testing()
    function(render_cpu_test name)
        add_executable(${name} ${ARGN})
        target_include_directories(${name} PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/src")
        target_link_libraries(${name} PRIVATE render::api glm::glm)
        set_target_properties(${name} PROPERTIES CXX_STANDARD 23 CXX_STANDARD_REQUIRED ON CXX_EXTENSIONS OFF)
        if(MSVC)
            target_compile_options(${name} PRIVATE /W4 /utf-8)
        endif()
        add_test(NAME ${name} COMMAND ${name})
    endfunction()
    render_cpu_test(render-mapping-tests tests/MappingTest.cpp src/MaterialMapping.cpp)
    render_cpu_test(render-dds-tests     tests/DdsDecodeTest.cpp src/DdsDecode.cpp)
    render_cpu_test(render-overlay-tests tests/OverlayGeometryTest.cpp src/OverlayGeometry.cpp)

    add_executable(render-gpu-tests tests/GpuRenderTest.cpp)
    target_link_libraries(render-gpu-tests PRIVATE render::diligent)
    set_target_properties(render-gpu-tests PROPERTIES CXX_STANDARD 23 CXX_STANDARD_REQUIRED ON CXX_EXTENSIONS OFF)
    if(MSVC)
        target_compile_options(render-gpu-tests PRIVATE /utf-8)
    endif()
    if(COMMAND copy_required_dlls)
        copy_required_dlls(render-gpu-tests)
    endif()
    foreach(_backend vulkan d3d12)
        add_test(NAME render-gpu-${_backend} COMMAND render-gpu-tests --backend ${_backend})
        set_tests_properties(render-gpu-${_backend} PROPERTIES LABELS gpu)
    endforeach()
endif()
