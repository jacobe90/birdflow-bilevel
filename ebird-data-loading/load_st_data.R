library(BirdFlowModels)
library(BirdFlowR)
library(terra)
library(ebirdst)

bf <- preprocess_species("amewoo", out_dir=".",hdf5 = TRUE, res=100)
