#!/bin/bash
# Benchmark device performance by compiling LLVM from source with a full parallel build.
set -e

DEVICE_TYPE=${1:-cpu}
WORK_DIR=${2:-$(mktemp -d)}
JOBS=${3:-$(nproc)}
LLVM_VERSION=${4:-llvmorg-19.1.0}

if ! command -v cmake &>/dev/null; then
    echo "Error: cmake is required but not installed." >&2
    exit 1
fi

if command -v ninja &>/dev/null; then
    GENERATOR="Ninja"
    BUILD_FLAGS="-j ${JOBS}"
elif command -v make &>/dev/null; then
    GENERATOR="Unix Makefiles"
    BUILD_FLAGS="-j ${JOBS}"
else
    echo "Error: neither ninja nor make is available." >&2
    exit 1
fi

echo "=== LLVM Compilation Benchmark ==="
echo "Device type: ${DEVICE_TYPE}"
echo "Working directory: ${WORK_DIR}"
echo "Parallel jobs: ${JOBS}"
echo "LLVM version: ${LLVM_VERSION}"
echo "CMake generator: ${GENERATOR}"
echo

mkdir -p "${WORK_DIR}"
cd "${WORK_DIR}"

if [ ! -d llvm-project ]; then
    echo "Cloning llvm-project ${LLVM_VERSION}..."
    git clone --depth 1 --branch "${LLVM_VERSION}" https://github.com/llvm/llvm-project.git
else
    echo "Using existing llvm-project directory"
fi

cd llvm-project

# Always start from a clean build directory so the benchmark measures a full build.
if [ -d build ]; then
    echo "Removing previous build directory for a clean benchmark run..."
    rm -rf build
fi

echo "Configuring LLVM..."
cmake -S llvm -B build -G "${GENERATOR}" \
    -DCMAKE_BUILD_TYPE=Release \
    -DLLVM_ENABLE_PROJECTS="clang" \
    -DLLVM_TARGETS_TO_BUILD="host" \
    -DLLVM_ENABLE_ASSERTIONS=OFF \
    -DLLVM_ENABLE_RTTI=OFF \
    -DLLVM_ENABLE_ZLIB=OFF \
    -DLLVM_ENABLE_TERMINFO=OFF \
    -DLLVM_ENABLE_LIBXML2=OFF \
    -DLLVM_ENABLE_LIBEDIT=OFF

echo
echo "Building LLVM with ${JOBS} parallel job(s)..."
START_TIME=$(date +%s)
START_ISO=$(date -Iseconds)

# shellcheck disable=SC2086
cmake --build build -- ${BUILD_FLAGS}

END_TIME=$(date +%s)
END_ISO=$(date -Iseconds)
ELAPSED=$((END_TIME - START_TIME))

echo
echo "=== LLVM Build Benchmark Result ==="
echo "device_type, llvm_version, jobs, elapsed_seconds, start_time, end_time"
echo "${DEVICE_TYPE}, ${LLVM_VERSION}, ${JOBS}, ${ELAPSED}, ${START_ISO}, ${END_ISO}"
