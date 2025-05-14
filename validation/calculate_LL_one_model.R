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

get_interval_obj <- function(hdf_root, species) {
  filenames <- list.files(path=hdf_root)
  hdf_path <- file.path(hdf_root, filenames[[1]]) # just use the first file in the list
  bf <- BirdFlowR::import_birdflow(hdf_path)
  species <- 'amewoo'
  params <- list(species = species)

  track_birdflowroutes_obj <- get_real_track(bf, params, filter=FALSE) # Real track. Not filtered by season. All year round.

  # Convert birdflowroutes object to birdflowintervals object
  interval_obj <- track_birdflowroutes_obj |>
    BirdFlowR::as_BirdFlowIntervals(max_n=20000, # the maximum number of intervals to extract
                                    min_day_interval=1,
                                    max_day_interval=365,
                                    min_km_interval=0,
                                    max_km_interval=8000)

}

get_log_likelihood_df <- function(hdf_root, interval_obj, validation_dir, w2_models=TRUE) {
  filenames <- list.files(path=hdf_root)

  grid_search_avg_ll_vals <- list()
  count <- 1
  for (hdf_name in filenames) {
    print(sprintf("Iteration %d of %d", count, length(filenames)))
    count <- count + 1

    hdf_path <- file.path(hdf_root, hdf_name) # get path to hdf

    # get birdflow object
    bf <- BirdFlowR::import_birdflow(hdf_path)

    # Get the LL
    ll_df <- get_interval_based_metrics(interval_obj, bf)

    # save ll_df to csv
    hyperparameters <- bf$metadata$hyperparameters
    interval_ll_filename <- sprintf("interval_lls_ow%f_ew%f_dw%f_dp%f.csv", hyperparameters$obs_weight, 
                                                      hyperparameters$ent_weight,
                                                      hyperparameters$dist_weight,
                                                      hyperparameters$dist_pow)
    interval_ll_filepath <- file.path(validation_dir, interval_ll_filename)
    write.csv(ll_df, interval_ll_filepath)
    
    # store hyperparameters and average log likelihood in array
    avg_ll <- mean(ll_df$ll)
    avg_null_ll <- mean(ll_df$null_ll)
    avg_ll_ci <- t.test(ll_df$ll)$conf # 95% confidence interval for avg ll
    avg_ll_ci_lower <- avg_ll_ci[[1]]
    avg_ll_ci_upper <- avg_ll_ci[[2]]
    null_ll_ci <- t.test(ll_df$null_ll)$conf # 95% confidence interval for null ll
    null_ll_ci_lower <- null_ll_ci[[1]]
    null_ll_ci_upper <- null_ll_ci[[2]]
    hyperparameters <- bf$metadata$hyperparameters
    ll_and_hyperparameters <- c(hyperparameters, list(avg_ll = avg_ll, 
                                                      avg_ll_conf_lower=avg_ll_ci_lower, 
                                                      avg_ll_conf_upper=avg_ll_ci_upper, 
                                                      avg_null_ll=avg_null_ll,
                                                      avg_null_ll_conf_lower=null_ll_ci_lower,
                                                      avg_null_ll_conf_upper=null_ll_ci_upper,
                                                      interval_ll_filepath=interval_ll_filepath))
    grid_search_avg_ll_vals <- append(grid_search_avg_ll_vals, list(ll_and_hyperparameters))
    print(ll_and_hyperparameters)
  }

  # get results df
  grid_search_results_df <- do.call(rbind, lapply(grid_search_avg_ll_vals, function(x) as.data.frame(x, stringsAsFactors = FALSE)))

  # write results to a csv
  if(w2_models) {
    write.csv(grid_search_results_df, file.path(validation_dir, "w2_grid_search_avg_lls.csv"))

  } else {
    write.csv(grid_search_results_df, file.path(validation_dir, "l2_grid_search_avg_lls.csv"))
  }
  
}

get_log_likelihood_df_one_model <- function(hdf_path, interval_obj, validation_dir, w2_models=TRUE) {
    # get birdflow object
    bf <- BirdFlowR::import_birdflow(hdf_path)

    # Get the LL
    ll_df <- get_interval_based_metrics(interval_obj, bf)

    # save ll_df to csv
    hyperparameters <- bf$metadata$hyperparameters
    if (w2_models) {
        interval_ll_filename <- sprintf("w2_interval_lls_ow%f_ew%f_dw%f_dp%f.csv", hyperparameters$obs_weight, 
                                                        hyperparameters$ent_weight,
                                                        hyperparameters$dist_weight,
                                                        hyperparameters$dist_pow)
    } else {
        interval_ll_filename <- sprintf("l2_interval_lls_ow%f_ew%f_dw%f_dp%f.csv", hyperparameters$obs_weight, 
                                                        hyperparameters$ent_weight,
                                                        hyperparameters$dist_weight,
                                                        hyperparameters$dist_pow)
    }
    interval_ll_filepath <- file.path(validation_dir, interval_ll_filename)
    write.csv(ll_df, interval_ll_filepath)
}

# get ll dfs for w2 and l2 grid searches
w2_best_hdf_path <- "/work/pi_drsheldon_umass_edu/birdflow_modeling/jacob_independent_study/birdflow-bilevel/experiment-results/w2-grid-search/hdfs/w2_53w_amewoo_2021_100km_obs0.75_ent0.014999999664723873_dist0.05999999865889549_pow0.25.hdf5"
l2_best_hdf_path <- "/work/pi_drsheldon_umass_edu/birdflow_modeling/jacob_independent_study/birdflow-bilevel/experiment-results/l2-grid-search/hdfs/amewoo_2021_100km_obs0.75_ent0.014999999664723873_dist0.05999999865889549_pow0.25.hdf5"
w2_hdf_root <- "/work/pi_drsheldon_umass_edu/birdflow_modeling/jacob_independent_study/birdflow-bilevel/experiment-results/w2-grid-search/hdfs"
validation_dir <- "/work/pi_drsheldon_umass_edu/birdflow_modeling/jacob_independent_study/birdflow-bilevel/experiment-results/amewoo-best-models"

# first compute interval obj
interval_obj <- get_interval_obj(w2_hdf_root, 'amewoo')

# save w2 / l2 log likelihood dfs
get_log_likelihood_df_one_model(w2_best_hdf_path, interval_obj, validation_dir)
get_log_likelihood_df_one_model(l2_best_hdf_path, interval_obj, validation_dir, w2_models=FALSE)
