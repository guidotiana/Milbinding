#!/bin/bash
mkdir datasets_and_RFs
cd datasets_and_RFs
export SERVER_URL=https://dataverse.unimi.it
export PERSISTENT_ID=doi:10.13130/RD_UNIMI/5879ZG

curl -L -O -J -H "X-Dataverse-key:$API_TOKEN" $SERVER_URL/api/access/dataset/:persistentId/?persistentId=$PERSISTENT_ID

unzip dataverse_files.zip
rm dataverse_files.zip
cd ..
