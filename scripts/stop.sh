#!/bin/bash
. "$(dirname "$0")/common.sh"
docker stop -t 30 "$CONTAINER" >/dev/null && docker rm "$CONTAINER" >/dev/null
log "stopped $CONTAINER"
