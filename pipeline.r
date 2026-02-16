run_pipeline <- function(sample_size = 1500, n_features = 5) {

  library(data.table)
  library(caret)
  library(ranger)
  library(pROC)
  library(jsonlite)

  set.seed(42)

  
  OUTPUT_DIR <- getwd()

 
  df <- fread("qsar_oral_toxicity.csv", sep = ";", header = FALSE)

  X <- df[, 1:(ncol(df) - 1)]
  y <- ifelse(df[[ncol(df)]] == "positive", 1, 0)


  pos_idx <- which(y == 1)
  neg_idx <- which(y == 0)

  n_pos <- round(sample_size * length(pos_idx) / length(y))
  n_neg <- sample_size - n_pos

  sub_idx <- c(
    sample(pos_idx, n_pos),
    sample(neg_idx, n_neg)
  )

  X_sub <- X[sub_idx, ]
  y_sub <- y[sub_idx]

 
  train_idx <- createDataPartition(y_sub, p = 0.8, list = FALSE)

  X_train <- X_sub[train_idx, ]
  X_test  <- X_sub[-train_idx, ]
  y_train <- y_sub[train_idx]
  y_test  <- y_sub[-train_idx]


  rf <- ranger(
    dependent.variable.name = "y",
    data = cbind(X_train, y = factor(y_train)),
    num.trees = 50,
    importance = "impurity"
  )

  top_features <- names(
    sort(rf$variable.importance, decreasing = TRUE)
  )[1:n_features]


  X_train_sel <- X_train[, ..top_features]
  X_test_sel  <- X_test[, ..top_features]

  scaler <- preProcess(X_train_sel, method = c("center", "scale"))
  X_train_scaled <- predict(scaler, X_train_sel)
  X_test_scaled  <- predict(scaler, X_test_sel)


  write.csv(
    X_train_scaled,
    file.path(OUTPUT_DIR, sprintf("X_train_quantum_%dfeatures.csv", n_features)),
    row.names = FALSE
  )

  write.csv(
    X_test_scaled,
    file.path(OUTPUT_DIR, sprintf("X_test_quantum_%dfeatures.csv", n_features)),
    row.names = FALSE
  )

  write.csv(
    data.frame(label = y_train),
    file.path(OUTPUT_DIR, sprintf("y_train_quantum_%dfeatures.csv", n_features)),
    row.names = FALSE
  )

  write.csv(
    data.frame(label = y_test),
    file.path(OUTPUT_DIR, sprintf("y_test_quantum_%dfeatures.csv", n_features)),
    row.names = FALSE
  )


  config <- list(
    num_features  = length(top_features),
    feature_names = top_features
  )

  write_json(
    config,
    file.path(OUTPUT_DIR, "quantum_config.json"),
    pretty = TRUE,
    auto_unbox = TRUE
  )

  cat("quantum_config.json written\n")


  cat("Running quantum pipeline (Python)...\n")

  old_wd <- getwd()
setwd(OUTPUT_DIR)

cat("Running quantum pipeline (Python)...\n")
system("python quantum_pipeline.py")

setwd(old_wd)

  return(list(
    features = top_features,
    train_size = length(y_train),
    test_size = length(y_test)
  ))
}
