# Docker files and images

Below is a table showing the Docker images you can create using the provided Docker files.
It is strongly suggested to follow the Docker image name suggestions to avoid confusion.

| Docker file | Docker image | Notes |
| ------ | --- | ----- |
| **briefly-base-dockerfile** | briefly-base | base image |
| **briefly-rest-dockerfile**  | briefly-rest  | briefly with REST i/f |
| **briefly-service-dockerfile** | briefly-service | briefly with choice of web & REST i/f |

## Build command for base image

First build the base image with the following command keeping `Briefly` as your
working directory:

`docker build -f briefly-base-dockerfile -t briefly-base:latest .`

This base image is used to build the other images.  Make sure you follow the
nomenclature suggested.  The base image is not meant to be run.

## Build command for service image

Build the service image with the following command after changing over to the
`Docker` (current) folder:

`sudo docker build -f briefly-service-dockerfile -t briefly-service:latest .`

## Docker run commands

Having built your Briefly service image, you can create containers running
either the web service or the REST service.  The services run on their respective
ports.

### Running the web service

`sudo docker run -p 7860:7860 briefly-service:latest`

### Running the REST service

`sudo docker run -p 8000:8000 briefly-service:latest REST`

For both the run commands, you can use the optional `-d` switch to detach the
service from the terminal.
