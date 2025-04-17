.libPaths(new = c("~/Rpackages", .libPaths()))

# install.packages('/work/pi_drsheldon_umass_edu/birdflow_modeling/jacob_independent_study/tmp/desirability2_0.0.1.tar.gz', repos = NULL, type="source")
# devtools::install_local("/work/pi_drsheldon_umass_edu/birdflow_modeling/jacob_independent_study/tmp/BirdFlowPipeline", force = T, dependencies = TRUE)
# devtools::install_local("/work/pi_drsheldon_umass_edu/birdflow_modeling/jacob_independent_study/tmp/BirdFlowR", force = T, dependencies = TRUE)

devtools::load_all("/work/pi_drsheldon_umass_edu/birdflow_modeling/jacob_independent_study/r-packages/BirdFlowPipeline")
devtools::load_all("/work/pi_drsheldon_umass_edu/birdflow_modeling/jacob_independent_study/r-packages/BirdFlowR")
# load_all("/home/yc85_illinois_edu/BirdFlowR") 

# Define functions
get_interval_based_validation_one_transition_pair <- function(birdflow_interval_row, bf, gcd, st_dists){
  
  # latlong data for banding and encounter location
  point_df_initial <- data.frame(x = birdflow_interval_row$lon1, y = birdflow_interval_row$lat1)
  point_df_final   <- data.frame(x = birdflow_interval_row$lon2, y = birdflow_interval_row$lat2)
  # birdflow one-hot distributions for banding and encounter locations
  d_initial <- as_distr(x = point_df_initial, bf = bf, crs = 'EPSG:4326') # same as birdflow_interval_row$i1
  d_final <- as_distr(x = point_df_final, bf = bf, crs = 'EPSG:4326') # same as birdflow_interval_row$i2
  # get s&t distribution for final timestep
  final_timestep <- birdflow_interval_row$timestep2
  final_st_distr <- st_dists[,final_timestep]
  # birdflow cell index for encounter location
  i_final <- which(d_final == 1)
  # birdflow predictions from banding one-hot, for encounter date
  preds <- predict(bf, d_initial, start = birdflow_interval_row$date1, end = birdflow_interval_row$date2)
  preds_final <- preds[,ncol(preds),drop = FALSE]
  preds_final <- as.vector(preds_final)
  # subset great circle distances for cell of actual encounter location
  gcd_final <- gcd[,i_final]
  # weighted average distance from predicted encounter distribution to actual encounter location
  
  # get location index of banding starting point
  loc_i_starting <- birdflow_interval_row$i1
  date_starting <- birdflow_interval_row$timestep1
  
  #
  elapsed_days <- as.numeric(birdflow_interval_row$date2 - birdflow_interval_row$date1, unit='days')
  elapsed_km <- great_circle_distance_lonlat_input(birdflow_interval_row$lat1, birdflow_interval_row$lon1,
                                                   birdflow_interval_row$lat2, birdflow_interval_row$lon2)
  
  # LL
  null_ll <- log(final_st_distr[i_final] + 1e-8)
  ll <- log(preds_final[i_final] + 1e-8)
  
  
  # return
  return(c(global_prob_of_the_starting = as.numeric(bf$distr[loc_i_starting,date_starting] / 52),
           elapsed_days = elapsed_days,
           elapsed_km = elapsed_km,
           null_ll = null_ll,
           ll = ll,
           lon1=birdflow_interval_row$lon1,
           lat1=birdflow_interval_row$lat1,
           lon2=birdflow_interval_row$lon2,
           lat2=birdflow_interval_row$lat2,
           date1=birdflow_interval_row$date1,
           date2=birdflow_interval_row$date2
  ))
}

get_interval_based_metrics <- function(birdflow_intervals, bf){
  # weekly distributions directly from S&T
  st_dists <- get_distr(bf, which = "all", from_marginals = FALSE)
  
  # Great circle distances between cells
  gcd <- great_circle_distances(bf)
  
  # Calculate ll
  dists <- sapply(split(birdflow_intervals$data, seq(nrow(birdflow_intervals$data))), get_interval_based_validation_one_transition_pair, bf, gcd, st_dists)
  dists <- t(dists)
  
  dists <- as.data.frame(dists)
  dists$date1 <- as.Date(dists$date1)
  dists$date2 <- as.Date(dists$date2)
  
  return(dists)
}


# Load your birdflow model here:
bf <- BirdFlowR::import_birdflow('/work/pi_drsheldon_umass_edu/birdflow_modeling/jacob_independent_study/birdflow-bilevel/experiment-results/w2_26w_amewoo_2021_100km_obs1.0_ent0.0001_dist0.01_pow0.4.hdf5')

# Load and track info
species <- 'amewoo'
params <- list(species = species)

track_birdflowroutes_obj <- get_real_track(bf, params, filter=FALSE) # Real track. Not filtered by season. All year round.

# Convert birdflowroutes object to birdflowintervals object
interval_obj <- track_birdflowroutes_obj |>
  BirdFlowR::as_BirdFlowIntervals(max_n=100, # the maximum number of intervals to extract
                                  min_day_interval=1,
                                  max_day_interval=180,
                                  min_km_interval=0,
                                  max_km_interval=8000)

# Get the LL
ll_df <- get_interval_based_metrics(interval_obj, bf)
head(ll_df)

