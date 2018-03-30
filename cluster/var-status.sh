#!/bin/sh
# Copyright 2018 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
set -e

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

echo "PROJECT: $PROJECT"
echo "LOGGING_PROJECT: $LOGGING_PROJECT"
echo "CLUSTER_NAME: $CLUSTER_NAME"
echo "BOARD_SIZE: $BOARD_SIZE"
echo "K8S_VERSION: $K8S_VERSION"
echo "ZONE: $ZONE"
echo "NUM_K8S_NODES: $NUM_K8S_NODES"

echo "SERVICE_ACCOUNT: $SERVICE_ACCOUNT"
echo "SERVICE_ACCOUNT_EMAIL: $SERVICE_ACCOUNT_EMAIL"
echo "SERVICE_ACCOUNT_KEY_LOCATION: $SERVICE_ACCOUNT_KEY_LOCATION"

echo "VERSION_TAG: $VERSION_TAG"
echo "GPU_PLAYER_CONTAINER: $GPU_PLAYER_CONTAINER"
echo "CPU_PLAYER_CONTAINER: $CPU_PLAYER_CONTAINER"

echo "BUCKET_NAME: $BUCKET_NAME"
echo "BUCKET_LOCATION: $BUCKET_LOCATION"
