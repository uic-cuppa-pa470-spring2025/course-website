## Load libraries

library(plumber)
library(tidyverse)

setwd("~/Documents/projects/course-website/week 10")

oecd_best_model <- readRDS("best_oecd_model.RDS")

#* This is a sample test case
#* @get /
function() {
  "Hello World"
}


#* This is my second page test
#* @get /second_page
function() {
  "This is my second page"
}

#* This endpoint adds 10 to the inputted number
#* @param num
#* @get /add_10
function(num) {
  as.numeric(num)+10
}

#* This endpoint multiplies two numbers
#* @param num_1 First number to multiply
#* @param num_2 Second number to multiply
#* @get /mul
function(num_1, num_2) {
  as.numeric(num_1) * as.numeric(num_2)
}






#* This endpoint returns the predicted value of age adjusted mortality rate from our random forest model
#* @param year TIME_PERIOD
#* @param forest_exposure Forest exposure to areas at risk of burning_Percentage of forested area
#* @param burnt_land Amount of burned area_Percentage of land area
#* @param risk_burn Population exposure to areas at risk of burning_Percentage of population
#* @param land_soil_moisture Land soil moisture anomaly_Percentage change
#* @param cropland_soil_moisture Cropland soil moisture anomaly_Percentage change
#* @param particulate_matter Fine particulate matter (PM2.5)_Microgrammes per cubic metre
#* @get /predict
function(year, forest_exposure, burnt_land, risk_burn, land_soil_moisture, cropland_soil_moisture, particulate_matter) {
  pred_new_data <- tibble(
    "REF_AREA" = c("NEW_AREA"),
    "TIME_PERIOD" = c(as.numeric(year)),
    "Forest exposure to areas at risk of burning_Percentage of forested area" = c(as.numeric(forest_exposure)),
    "Amount of burned area_Percentage of land area" = c(as.numeric(burnt_land)),
    "Population exposure to areas at risk of burning_Percentage of population" = c(as.numeric(risk_burn)),
    "Land soil moisture anomaly_Percentage change" = c(as.numeric(land_soil_moisture)),
    "Cropland soil moisture anomaly_Percentage change" = c(as.numeric(cropland_soil_moisture)),
    "Fine particulate matter (PM2.5)_Microgrammes per cubic metre" = c(as.numeric(particulate_matter)),
  )
  
   %>% predict(pred_new_data)
  
}



