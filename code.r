# Load necessary libraries
library(tidymodels)

# Step 1: Load your data
# Assuming you have your data loaded into a data frame named 'data'
# Replace 'data' with your actual data frame
data <- read.csv("your_data.csv")

# Step 2: Create a recipe for data preprocessing
data_recipe <- 
  recipe(target_variable ~ ., data = data) %>%
  step_dummy(all_nominal(), -all_outcomes()) %>%
  step_zv(all_predictors()) %>%
  step_nzv(all_predictors()) %>%
  step_normalize(all_numeric_predictors())

# Step 3: Create a model specification
xgb_spec <- 
  boost_tree(
    mode = "classification",  # Classification mode
    trees = 100,              # Number of trees (you can adjust this)
    mtry = 10,                # Number of variables to sample at each split (you can adjust this)
    learn_rate = 0.1          # Learning rate (you can adjust this)
  ) %>%
  set_engine("xgboost")

# Step 4: Create a resampling method (cross-validation)
cv <- vfold_cv(data, v = 10)  # 10-fold cross-validation

# Step 5: Create a workflow
xgb_workflow <- 
  workflow() %>%
  add_recipe(data_recipe) %>%
  add_model(xgb_spec)

# Step 6: Train and evaluate the model
xgb_results <- 
  fit_resamples(
    xgb_workflow, 
    resamples = cv,
    metrics = metric_set(roc_auc, accuracy)  # Classification metrics
  )

# Step 7: Summarize the results
xgb_results %>%
  collect_metrics() %>%
  summarise_all(list(mean = mean, sd = sd))  # Summary statistics for metrics

# Step 8: Optionally, you can also get the final model for predictions
final_xgb_model <- 
  finalize_model(xgb_workflow)

# Step 9: Make predictions with the final model (if needed)
# Replace 'new_data' with your new data for predictions
# predictions <- predict(final_xgb_model, new_data)

# Step 10: Optionally, save the final model for future use
# saveRDS(final_xgb_model, "xgb_classification_model.rds")
