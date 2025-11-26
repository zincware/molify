#!/bin/bash

docker run --rm  --volume $PWD:/data --user $(id -u):$(id -g) --platform=linux/amd64 --env JOURNAL=joss openjournals/inara
