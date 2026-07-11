#!/bin/sh
if [ -z "${BASH_VERSION:-}" ]; then
  apk add --no-cache bash >/dev/null
  exec bash "$0" "$@"
fi
set -euo pipefail

: "${PRIME_COMMIT:?PRIME_COMMIT must pin the isolated Prime integration branch}"
: "${DYNAMO_COMMIT:?DYNAMO_COMMIT must pin the isolated Dynamo sidecar branch}"
: "${VLLM_COMMIT:?VLLM_COMMIT must pin the local vLLM OpenEngine branch}"
: "${VLLM_BUNDLE_SHA256:?VLLM_BUNDLE_SHA256 must pin the transferred Git bundle}"
: "${GITEA_USERNAME:?GITEA_USERNAME must be injected from a Kubernetes Secret}"
: "${GITEA_PASSWORD:?GITEA_PASSWORD must be injected from a Kubernetes Secret}"

PRIME_BRANCH=${PRIME_BRANCH:-bis/openengine-rl-k8s}
DYNAMO_BRANCH=${DYNAMO_BRANCH:-bis/openengine-rl-dynamo-v2}
VLLM_BASE_COMMIT=${VLLM_BASE_COMMIT:-1bd8f80a643efb5f977469692e458fd9a4fd3b4f}
VLLM_OPENENGINE_SERVICE_COMMIT=${VLLM_OPENENGINE_SERVICE_COMMIT:-052f0bacd4dd9012e3ac647554c5c7cdca296a9c}
OPENENGINE_PROTO_SHA256=${OPENENGINE_PROTO_SHA256:-ad7be818be967084c9ae47ba9db526810fd0020611bc5b6a00b6016601ba67b2}
PRIME_RL_PROTO_SHA256=${PRIME_RL_PROTO_SHA256:-c56d5e41e9bf599cca64af2f70b8fc782580f218dffe39ee81bef5ab23737f99}
VLLM_BUNDLE=${VLLM_BUNDLE:-/build/incoming/vllm-openengine.bundle}
PRIME_BUNDLE=${PRIME_BUNDLE:-}
PRIME_BUNDLE_SHA256=${PRIME_BUNDLE_SHA256:-}
DYNAMO_BUNDLE=${DYNAMO_BUNDLE:-}
DYNAMO_BUNDLE_SHA256=${DYNAMO_BUNDLE_SHA256:-}
GITEA_URL=${GITEA_URL:-http://biswa-gitea:3000}
IMAGE_REPOSITORY=${IMAGE_REPOSITORY:-nvcr.io/nvidian/dynamo-dev/biswa}
BASE_IMAGE=${BASE_IMAGE:-nvcr.io/nvidian/dynamo-dev/biswa:prime-p4-39bc54c8096c-dynamo-f4ac89d6eac0-arm64@sha256:07bb41622c701743e9497246ebc97e2374b1f5265bf1897bcba0d96201a2d355}
RUN_TS=${RUN_TS:-$(date -u +%Y%m%dT%H%M%SZ)}
WORK_ROOT=/build/openengine-$RUN_TS
ARTIFACT_ROOT=/models/bis-rl-3/biswa-p4/builds/openengine-$RUN_TS
DOCKER_CONFIG=$WORK_ROOT/docker-config
export DOCKER_CONFIG

cleanup() {
  rm -f /tmp/gitea-askpass
  rm -rf "$DOCKER_CONFIG"
}
trap cleanup EXIT
trap 'exit 129' HUP
trap 'exit 130' INT
trap 'exit 143' TERM

mkdir -p "$DOCKER_CONFIG" "$WORK_ROOT" "$ARTIFACT_ROOT"
cp /run/secrets/nvcr/config.json "$DOCKER_CONFIG/config.json"
apk add --no-cache bash coreutils git jq >/dev/null

cat > /tmp/gitea-askpass <<'EOF'
#!/bin/sh
case "$1" in
  *Username*) printf '%s\n' "$GITEA_USERNAME" ;;
  *Password*) printf '%s\n' "$GITEA_PASSWORD" ;;
  *) exit 1 ;;
esac
EOF
chmod 700 /tmp/gitea-askpass
export GIT_ASKPASS=/tmp/gitea-askpass
export GIT_TERMINAL_PROMPT=0

clone_exact() {
  local repository=$1 branch=$2 commit=$3 destination=$4 bundle=$5 bundle_sha256=$6
  if [ -n "$bundle" ]; then
    test -n "$bundle_sha256"
    test "$(sha256sum "$bundle" | awk '{print $1}')" = "$bundle_sha256"
    GIT_LFS_SKIP_SMUDGE=1 git clone "$bundle" "$destination"
    git -C "$destination" checkout --detach "$commit"
  else
    GIT_LFS_SKIP_SMUDGE=1 git clone --branch "$branch" --single-branch \
      "$GITEA_URL/biswa/$repository.git" "$destination"
  fi
  test "$(git -C "$destination" rev-parse HEAD)" = "$commit"
  test -z "$(git -C "$destination" status --porcelain)"
}

clone_exact prime-rl-2 "$PRIME_BRANCH" "$PRIME_COMMIT" "$WORK_ROOT/prime-rl" \
  "$PRIME_BUNDLE" "$PRIME_BUNDLE_SHA256"
clone_exact dynamo2 "$DYNAMO_BRANCH" "$DYNAMO_COMMIT" "$WORK_ROOT/dynamo" \
  "$DYNAMO_BUNDLE" "$DYNAMO_BUNDLE_SHA256"
test "$(sha256sum "$VLLM_BUNDLE" | awk '{print $1}')" = "$VLLM_BUNDLE_SHA256"
GIT_LFS_SKIP_SMUDGE=1 git clone "$VLLM_BUNDLE" "$WORK_ROOT/vllm"
git -C "$WORK_ROOT/vllm" checkout --detach "$VLLM_COMMIT"
test "$(git -C "$WORK_ROOT/vllm" rev-parse HEAD)" = "$VLLM_COMMIT"
test -z "$(git -C "$WORK_ROOT/vllm" status --porcelain)"
test "$(git -C "$WORK_ROOT/vllm" merge-base "$VLLM_COMMIT" "$VLLM_OPENENGINE_SERVICE_COMMIT")" = \
  "$VLLM_OPENENGINE_SERVICE_COMMIT"

proto_sha=$(sha256sum "$WORK_ROOT/vllm/rust/proto/openengine.proto" | awk '{print $1}')
test "$proto_sha" = "$OPENENGINE_PROTO_SHA256"
cmp "$WORK_ROOT/vllm/rust/proto/openengine.proto" \
  "$WORK_ROOT/dynamo/lib/vllm-sidecar/proto/openengine.proto"
prime_rl_proto_sha=$(sha256sum "$WORK_ROOT/vllm/rust/proto/prime_rl.proto" | awk '{print $1}')
test "$prime_rl_proto_sha" = "$PRIME_RL_PROTO_SHA256"
cmp "$WORK_ROOT/vllm/rust/proto/prime_rl.proto" \
  "$WORK_ROOT/dynamo/lib/vllm-sidecar/proto/prime_rl.proto"

prime_short=${PRIME_COMMIT:0:12}
dynamo_short=${DYNAMO_COMMIT:0:12}
vllm_short=${VLLM_COMMIT:0:12}
image="$IMAGE_REPOSITORY:prime-${prime_short}-dynamo-${dynamo_short}-vllm-${vllm_short}-openengine-arm64"

cp "$WORK_ROOT/prime-rl/k8s/openengine/Dockerfile.arm64" "$WORK_ROOT/Dockerfile"
cat > "$ARTIFACT_ROOT/provenance.env" <<EOF
RUN_TS=$RUN_TS
BASE_IMAGE=$BASE_IMAGE
PRIME_BRANCH=$PRIME_BRANCH
PRIME_COMMIT=$PRIME_COMMIT
PRIME_BUNDLE=$PRIME_BUNDLE
PRIME_BUNDLE_SHA256=$PRIME_BUNDLE_SHA256
DYNAMO_BRANCH=$DYNAMO_BRANCH
DYNAMO_COMMIT=$DYNAMO_COMMIT
DYNAMO_BUNDLE=$DYNAMO_BUNDLE
DYNAMO_BUNDLE_SHA256=$DYNAMO_BUNDLE_SHA256
VLLM_COMMIT=$VLLM_COMMIT
VLLM_BASE_COMMIT=$VLLM_BASE_COMMIT
VLLM_OPENENGINE_SERVICE_COMMIT=$VLLM_OPENENGINE_SERVICE_COMMIT
VLLM_BUNDLE=$VLLM_BUNDLE
VLLM_BUNDLE_SHA256=$VLLM_BUNDLE_SHA256
OPENENGINE_PROTO_SHA256=$OPENENGINE_PROTO_SHA256
PRIME_RL_PROTO_SHA256=$PRIME_RL_PROTO_SHA256
IMAGE=$image
PLATFORM=linux/arm64
EOF
sha256sum "$WORK_ROOT/Dockerfile" > "$ARTIFACT_ROOT/Dockerfile.arm64.sha256"
sha256sum "$0" > "$ARTIFACT_ROOT/build-in-dind.sh.sha256"
git -C "$WORK_ROOT/prime-rl" status --short --branch > "$ARTIFACT_ROOT/prime-status.txt"
git -C "$WORK_ROOT/dynamo" status --short --branch > "$ARTIFACT_ROOT/dynamo-status.txt"
git -C "$WORK_ROOT/vllm" status --short --branch > "$ARTIFACT_ROOT/vllm-status.txt"

DOCKER_BUILDKIT=1 docker buildx build \
  --platform linux/arm64 \
  --file "$WORK_ROOT/Dockerfile" \
  --build-arg BASE_IMAGE="$BASE_IMAGE" \
  --build-arg PRIME_COMMIT="$PRIME_COMMIT" \
  --build-arg DYNAMO_COMMIT="$DYNAMO_COMMIT" \
  --build-arg VLLM_COMMIT="$VLLM_COMMIT" \
  --build-arg VLLM_BASE_COMMIT="$VLLM_BASE_COMMIT" \
  --build-arg VLLM_OPENENGINE_SERVICE_COMMIT="$VLLM_OPENENGINE_SERVICE_COMMIT" \
  --build-arg OPENENGINE_PROTO_SHA256="$OPENENGINE_PROTO_SHA256" \
  --build-arg PRIME_RL_PROTO_SHA256="$PRIME_RL_PROTO_SHA256" \
  --tag "$image" \
  --push \
  --progress plain \
  --metadata-file "$ARTIFACT_ROOT/build-metadata.json" \
  "$WORK_ROOT" 2>&1 | tee "$ARTIFACT_ROOT/build.log"

docker buildx imagetools inspect "$image" > "$ARTIFACT_ROOT/image-inspect.txt"
digest=$(jq -r '."containerimage.digest"' "$ARTIFACT_ROOT/build-metadata.json")
case "$digest" in
  sha256:*) ;;
  *) echo "missing image digest" >&2; exit 1 ;;
esac
image_ref="$image@$digest"
printf 'IMAGE_DIGEST=%s\nIMAGE_REF=%s\n' "$digest" "$image_ref" \
  >> "$ARTIFACT_ROOT/provenance.env"
printf '%s\n' "$image_ref" > "$ARTIFACT_ROOT/FINAL_IMAGE"
printf '%s\n' "$ARTIFACT_ROOT" > /models/bis-rl-3/biswa-p4/builds/CURRENT_OPENENGINE_BUILD
printf '%s\n' "$image_ref" > /models/bis-rl-3/biswa-p4/builds/CURRENT_OPENENGINE_IMAGE

(
  cd "$ARTIFACT_ROOT"
  find . -maxdepth 1 -type f \
    ! -name artifact-manifest.sha256 \
    ! -name driver.log \
    -print0 | sort -z | xargs -0 sha256sum
) > "$ARTIFACT_ROOT/artifact-manifest.sha256"

printf '%s\n' "$image_ref"
