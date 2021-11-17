library(lime)
library(tidyquant)
library(rsample)
library(recipes)
library(yardstick)
library(corrr)
library(readr)
library(ggplot2)
library(forcats)

churn_data_raw <- read_csv("data/WA_Fn-UseC_-Telco-Customer-Churn.csv")

# Remove unnecessary data
churn_data_tbl <- churn_data_raw %>%
  tidyr::drop_na() %>%
  select(Churn, everything())

# Split test/training sets
set.seed(100)
train_test_split <- initial_split(churn_data_tbl, prop = 0.8)
train_test_split


# Retrieve train and test sets
train_tbl_with_ids <- training(train_test_split)
test_tbl_with_ids  <- testing(train_test_split)

train_tbl <- select(train_tbl_with_ids, -customerID)
test_tbl <- select(test_tbl_with_ids, -customerID)

# Determine if log transformation improves correlation 
# between TotalCharges and Churn
train_tbl %>%
  select(Churn, TotalCharges) %>%
  mutate(
    Churn = Churn %>% as.factor() %>% as.numeric(),
    LogTotalCharges = log(TotalCharges)
  ) %>%
  correlate() %>%
  focus(Churn) %>%
  fashion()

# Create recipe
rec_obj <- recipe(Churn ~ ., data = train_tbl) %>%
  step_discretize(tenure, options = list(cuts = 6)) %>%
  step_log(TotalCharges) %>%
  step_dummy(all_nominal(), -all_outcomes()) %>%
  step_center(all_predictors(), -all_outcomes()) %>%
  step_scale(all_predictors(), -all_outcomes()) %>%
  prep(data = train_tbl)

x_train_tbl <- bake(rec_obj, new_data = train_tbl) %>% 
  mutate(Churn = as.factor(ifelse(Churn == "Yes", 1, 0)))

x_test_tbl  <- bake(rec_obj, new_data = test_tbl) %>% 
  mutate(Churn = as.factor(ifelse(Churn == "Yes", 1, 0)))



library(h2o)

h2o.init()


response <- "Churn"
predictors <- setdiff(colnames(x_train_tbl), "Opened")


gbm <- h2o.gbm(y = response,
                    x = predictors,
                    training_frame = as.h2o(x_train_tbl),
                    seed = 1)



arrests_pred <- h2o.predict(gbm, newdata = as.h2o(x_test_tbl))



h2o.partialPlot(gbm, data = as.h2o(x_test_tbl), cols = colnames(x_test_tbl))


explainer_h2o_gbm <- lime(x_test_tbl, gbm, n_bins = 5)

local_obs <- x_test_tbl[c(1, 20, 300),]


explanation_gbm <- explain(local_obs, 
                           explainer_h2o_gbm, 
                           n_features = 5,
                           labels = 1,
                           feature_select = "highest_weights")

p3 <- 
  plot_features(explanation_gbm, ncol = 1) + ggtitle("gbm")


p3





h2o.saveModel(gbm, path = getwd(), force = TRUE)

write.csv(x_test_tbl, 'x_test_tbl.csv', row.names = F)







