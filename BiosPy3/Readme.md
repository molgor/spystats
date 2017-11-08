# A containerised environment for Spystats and Biospytial

**Juan Escamilla Molgora <j.escamillamolgora@lancaster.ac.uk>**

** November 8, 2017 **

## Requirements:
* Docker engine
* Docker compose

## Building
For building the image do:
   
     docker build -t molgor/biospytial3:automatic .

Once built, for accessing to the container do:

     docker-compose up -d

**YOu need to be in the same directory as the docker-compose.yml file.**



