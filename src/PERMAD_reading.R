library(pacman)
pacman::p_load(tidyverse, readxl, openxlsx, zeallot, stringr)
setwd(dirname(rstudioapi::getActiveDocumentContext()$path))

load("data/PERMAD/PERMAD_Kohorte1+2_ct.RData")

data <- dataset$data
labs <- dataset$labs
time <- dataset$time
sample_id <- dataset$sample.id
features <- dataset$features
cohort <- dataset$cohort
time_id <- dataset$time.id
time_ct <- dataset$time.ct


max(table(dataset$sample.id))

save(data, file = "data.RData")
save(labs, file = "labs.RData")
save(time, file = "time.RData")
save(sample_id, file = "sample_id.RData")
save(sample_id, file = "sample_id.RData")
save(cohort, file = "cohort.RData")
save(time_id, file = "time_id.RData")
save(time_ct, file = "time_ct.RData")
save(features, file = "features.RData")
