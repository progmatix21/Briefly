#!/bin/bash
#script to select REST or web service.

if [ "$1" = "" ]; then
	echo "Starting web service."
	/usr/local/bin/python3 briefly.py
	
elif [ "$1" = "REST" ]; then
	echo "Starting REST service."
	fastapi run briefly.py
	
else
	echo "Invalid option, exiting"
fi

