#!/bin/bash

if [ -d "myenv" ]
then
    echo "python virtual environment exists."
else
    python3 -m venv myenv
fi

source myenv/bin/activate


sudo pip3 install -r requirements.txt

if [ -d "logs" ]
then
    echo "Log folder exists."
else
    mkdir logs
    touch logs/error.log logs/access.log
fi

