cmake_minimum_required(VERSION 3.10)
project(FinancialParallelComputing)

# Specify the C++ standard
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED True)

# Find OpenCL package
find_package(OpenCL REQUIRED)

# Add the executable
add_executable(financial_parallel_computing main.cpp)

# Include directories
target_include_directories(financial_parallel_computing PRIVATE ${OpenCL_INCLUDE_DIRS})

# Link OpenCL library
target_link_libraries(financial_parallel_computing PRIVATE ${OpenCL_LIBRARIES})

# Add the OpenCL kernels
set(OPENCL_KERNELS
    knl_source_pso_computeCosts.c
    knl_source_pso_searchGrid.c
    knl_src_pre_calc_ClassicAdjoint.c
    knl_src_pre_calc_GaussJordan.c
)

# Copy kernels to the build directory
foreach(KERNEL ${OPENCL_KERNELS})
    add_custom_command(
        TARGET financial_parallel_computing POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy
            ${CMAKE_SOURCE_DIR}/${KERNEL}
            $<TARGET_FILE_DIR:financial_parallel_computing>
    )
endforeach()