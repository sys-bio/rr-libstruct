CMAKE_MINIMUM_REQUIRED(VERSION 3.14)
project(rr-libstruct)

if (${CMAKE_VERSION} VERSION_GREATER_EQUAL "3.20")
    # Source file extensions do not need to be explicitly
    #   specified if CMP0115 set to OLD.
    #   New behavior requires all extensions be explicit
    cmake_policy(SET CMP0115 OLD)
endif()

# Option that allows users to build static and/or shared libraries
# ----------------------------------------------------------------
OPTION(LIBSTRUCT_BUILD_STATIC_LIBS "Build static libraries" ON)
OPTION(LIBSTRUCT_BUILD_SHARED_LIBS "Build shared libraries" OFF)

get_filename_component(THIRD_PARTY_DIR ${CMAKE_CURRENT_SOURCE_DIR} DIRECTORY)

#set(LIBSBML_INSTALL_PREFIX ${THIRD_PARTY_DIR}/libSBML-5.18.1-experimental-Source/install-${CMAKE_CXX_COMPILER_ID}
#        CACHE PATH "Path to the root of libsbml installation")
#set(CLAPACK_INSTALL_PREFIX ${THIRD_PARTY_DIR}/clapack3.2.1/install-${CMAKE_CXX_COMPILER_ID}
#        CACHE PATH "Path to the root of clapack installation")

set(CMAKE_POSITION_INDEPENDENT_CODE ON)

set(target rr-libstruct)

set(sources
        lsMatrix
        lsLibStructural
        lsSBMLModel
        lsUtils
        lsLibla
        lsLA
        lsLUResult
        )


#Then the static version
add_library(${target}-static STATIC ${sources})

set(static_compile_definitions
        STATIC_LIBSTRUCT
        LIBSBML_USE_CPP_NAMESPACE
        LIBSBML_STATIC
        )

if (WIN32)
    set(static_compile_definitions
            ${static_compile_definitions}
            WIN32
            )
endif ()

target_include_directories(${target}-static PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}
        ${CLAPACK_INCLUDE_DIR}
        ${LIBSBML_INCLUDE_DIR}
        )

target_link_libraries(${target}-static PRIVATE ${LIBSBML_STATIC_LIBRARY})


set_property(TARGET ${target}-static
        PROPERTY COMPILE_DEFINITIONS
        ${static_compile_definitions}
        )

#libraries to link with
if (${MINGW})
else ()
    #target_link_libraries (${target} libf2c)
    #target_link_libraries (${target}-static libf2c)
endif ()

include(GNUInstallDirs)

set(headers
        lsExporter.h
        lsMatrix.h
        lsLibStructural.h
        lsLA.h
        lsLibla.h
        lsLUResult.h
        lsUtils.h
        )

install(TARGETS ${target}-static
        EXPORT ${target}
        RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
        LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
        ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
        PUBLIC_HEADER DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/${target}
        )

# install the export info
install(
        EXPORT ${target}
        DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake
        NAMESPACE ${target}::
        FILE ${target}-config.cmake
)

#install public headers

message(STATUS "${CMAKE_INSTALL_INCLUDEDIR}/rr-libstruct")
install(FILES ${headers} DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}/rr-libstruct")

