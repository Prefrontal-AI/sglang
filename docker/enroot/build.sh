#!/bin/bash
# Build the serving image of one commit of this checkout and write it as an enroot squashfs.
#
#   bash docker/enroot/build.sh [--ref <commit>] [--layer <dir>] [--cuda <version>] [--out-dir <dir>] [--name <name>]
#
# Run from inside the checkout. The commit's tree (git archive, so uncommitted changes never leak in) builds
# docker/Dockerfile's framework_final stage; --layer puts <dir>/Dockerfile from the same tree on top, with
# BASE_IMAGE set to it. The result is <out-dir>/<name>-<sha10>.sqsh. Needs no GPU. Docker's store is per node,
# so the import runs where the build did, and only this build's own images and scratch are removed afterwards.
# Needs Docker with buildx and enroot. The default out-dir is ./images under the checkout, set by --out-dir or SGLANG_IMAGE_DIR.
# The scratch dir is $TMPDIR (default /tmp), set by SGLANG_BUILD_SCRATCH, and must hold the source tree and enroot's cache.
# Under Slurm, submit it with sbatch and your own resource flags - a build wants many cores and ~200G of memory.
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"
cd "$(git rev-parse --show-toplevel)"

ref=HEAD
layer=
cuda=13.0.3
out_dir=${SGLANG_IMAGE_DIR:-./images}
name=
while [ $# -gt 0 ]; do
    case $1 in
        --ref) ref=$2; shift 2 ;;
        --layer) layer=${2%/}; shift 2 ;;
        --cuda) cuda=$2; shift 2 ;;
        --out-dir) out_dir=$2; shift 2 ;;
        --name) name=$2; shift 2 ;;
        *) echo "unknown argument $1" >&2; exit 2 ;;
    esac
done
sha=$(git rev-parse --verify "$ref^{commit}")
short=${sha:0:10}
name=${name:-sglang-lsa${layer:+-$(basename "$layer")}}
out=$out_dir/$name-$short.sqsh
base_tag=sglang-fork-base:$short
tag=$name:$short

scratch_root=${SGLANG_BUILD_SCRATCH:-${TMPDIR:-/tmp}}
mkdir -p "$scratch_root"
scratch=$(mktemp -d "$scratch_root/sglang-build-$short.XXXXXX")
cleanup() {
    docker ps -a -q --filter "ancestor=$tag" | xargs -r docker rm >/dev/null
    docker rmi "$tag" "$base_tag" >/dev/null 2>&1 || true
    rm -rf "$scratch"
}
trap cleanup EXIT

mkdir -p "$scratch/src" "$out_dir"
git archive "$sha" | tar -x -C "$scratch/src"
if [ -n "$layer" ] && [ ! -f "$scratch/src/$layer/Dockerfile" ]; then
    echo "no $layer/Dockerfile at $short" >&2
    exit 1
fi
# An archive has no git history, so setuptools-scm falls back to its fallback version
# => stamp the commit there, so a server reports which build it is
sed -i "s/^fallback_version = \"0.0.0.dev0\"$/fallback_version = \"0.0.0.dev0+g$short\"/" "$scratch/src/python/pyproject.toml"
grep -q "^fallback_version = \"0.0.0.dev0+g$short\"$" "$scratch/src/python/pyproject.toml"

echo "building $sha as $tag on $(hostname), $(date -u)"
docker buildx build --target framework_final --platform linux/amd64 --load \
    -f "$scratch/src/docker/Dockerfile" \
    --build-arg CUDA_VERSION="$cuda" \
    --build-arg BUILD_TYPE=all \
    --build-arg BRANCH_TYPE=local \
    --build-arg INSTALL_FLASHINFER_JIT_CACHE=1 \
    --build-arg CMAKE_BUILD_PARALLEL_LEVEL="$(nproc)" \
    -t "$base_tag" "$scratch/src"
if [ -n "$layer" ]; then
    docker buildx build --platform linux/amd64 --load \
        -f "$scratch/src/$layer/Dockerfile" \
        --build-arg BASE_IMAGE="$base_tag" \
        -t "$tag" "$scratch/src"
else
    docker tag "$base_tag" "$tag"
fi

export ENROOT_CACHE_PATH=$scratch/enroot-cache ENROOT_DATA_PATH=$scratch/enroot-data
enroot import -o "$out.partial" "dockerd://$tag"
mv "$out.partial" "$out"
echo "squashfs $out, $(date -u)"
