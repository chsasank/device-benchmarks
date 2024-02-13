#!/bin/bash
# https://stackoverflow.com/a/34676160
set -e
DEVICE_TYPE=${1:-cpu}

# the directory of the script
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

# the temp directory used, within $DIR
# omit the -p parameter to create a temporal directory in the default location
WORK_DIR=`mktemp -d -p "$DIR"`

# check if tmp dir was created
if [[ ! "$WORK_DIR" || ! -d "$WORK_DIR" ]]; then
  echo "Could not create temp dir"
  exit 1
fi

# deletes the temp directory
function cleanup {      
  rm -rf "$WORK_DIR"
  echo "Deleted temp working directory $WORK_DIR"
}

# register the cleanup function to be called on the EXIT signal
trap cleanup EXIT

# implementation of script starts here
mkdir -p models
echo "Downloading mistral 7b int4 model from hugging face"
wget -nc https://huggingface.co/TheBloke/Mistral-7B-v0.1-GGUF/resolve/main/mistral-7b-v0.1.Q4_0.gguf -P models

cd $WORK_DIR
git clone https://github.com/ggerganov/llama.cpp
cd llama.cpp
mkdir build
cd build

case $DEVICE_TYPE in
  cpu)
    echo -n "Building for CPU"
    cmake ..
    ;;

  cuda)
    echo -n "Using CuBlas (Nvidia)"
    cmake .. -DLLAMA_CUBLAS=ON
    ;;

  rocm)
    echo -n "Using Rocm (AMD)"
    AMD_TARGET=$(rocminfo | grep gfx | head -1 | awk '{print $2}')
    AMD_TARGET=${AMD_TARGET%?}0
    export CC=/opt/rocm/llvm/bin/clang
    export CXX=/opt/rocm/llvm/bin/clang++
    cmake -DLLAMA_HIPBLAS=ON -DAMDGPU_TARGETS=gfx1100 -DCMAKE_BUILD_TYPE=Release ..
    ;;

  sycl)
    echo -n "Using Sycl (Intel)"
    cmake .. -DLLAMA_SYCL=ON -DCMAKE_C_COMPILER=icx -DCMAKE_CXX_COMPILER=icpx
    ;;

  *)
    echo -n "unknown"
    exit
    ;;
esac

cmake --build . --config Release -v -j
cd ..

./build/bin/llama-bench -m $DIR/models/mistral-7b-v0.1.Q4_0.gguf -p 128,256,512 -n 128,256,512