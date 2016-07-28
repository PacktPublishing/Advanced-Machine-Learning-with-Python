#!/bin/sh

CP="tagger.jar:dep/jackson-core-2.0.5.jar:dep/jackson-annotations-2.0.5.jar:dep/jackson-databind-2.0.5.jar:dep/stanford-parser.jar:dep/stanford-parser-2012-07-09-models.jar:dep/stanford-postagger.jar"

java -cp $CP Tagger $@
