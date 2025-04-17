## Load libraries

library(plumber)

pr("self_api.R") %>%
  pr_run(port=8000)