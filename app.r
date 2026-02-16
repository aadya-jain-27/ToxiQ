library(shiny)
library(caret)
library(data.table)


source("pipeline.R")

ui <- fluidPage(
  titlePanel("Quantum ML Drug Safety Screening"),
  
  sidebarLayout(
    sidebarPanel(
      sliderInput("sample_size", "Sample size:", min = 500, max = 1500, value = 1500, step = 100),
      sliderInput("n_features", "Number of features / qubits:", min = 3, max = 8, value = 5, step = 1),
      actionButton("run_pipeline", "Run Pipeline"),
      br(), br(),
      fileInput("user_csv", "Upload your own dataset (optional)", accept = ".csv")
    ),
    
    mainPanel(
      verbatimTextOutput("status"),
      h4("Top Features Selected:"),
      tableOutput("top_features"),
      h4("Quantum / Classical Results Preview:"),
      tableOutput("results_preview")
    )
  )
)

server <- function(input, output, session) {
  
  pipeline_result <- reactiveVal(NULL)
  
  observeEvent(input$run_pipeline, {
    output$status <- renderText("Running R pipeline...")
    
  
    result <- run_pipeline(sample_size = input$sample_size,
                           n_features = input$n_features)
    
    pipeline_result(result)  # store result
    
    output$status <- renderText(
      paste0("R Pipeline complete!\nTrain size: ", result$train_size,
             "\nTest size: ", result$test_size)
    )
    
    output$top_features <- renderTable({
      data.frame(Feature = result$features)
    })
    
    quantum_file <- "quantum_results.csv"
    if (file.exists(quantum_file)) {
      output$results_preview <- renderTable({
        head(read.csv(quantum_file))
      })
    } else {
      output$results_preview <- renderTable({
        data.frame(Message = "Quantum results not yet available")
      })
    }
  })
  
  observeEvent(input$user_csv, {
    req(input$user_csv)
    
    if (is.null(pipeline_result())) {
      output$status <- renderText("Please run the pipeline first!")
      return()
    }
   
    user_data <- read.csv(input$user_csv$datapath)
    
   
    top_features <- pipeline_result()$features
    missing_feats <- setdiff(top_features, colnames(user_data))
    if (length(missing_feats) > 0) {
      output$status <- renderText(
        paste("Uploaded data is missing these top features:", paste(missing_feats, collapse=", "))
      )
      return()
    }
    
    user_subset <- user_data[, top_features, drop=FALSE]
    
    
    X_train_scaled_file <- sprintf("X_train_quantum_%dfeatures.csv", length(top_features))
    X_train_scaled <- read.csv(X_train_scaled_file)
    
    scaler <- preProcess(X_train_scaled, method = c("center", "scale"))
    user_scaled <- predict(scaler, user_subset)
    
    
    y_train_file <- sprintf("y_train_quantum_%dfeatures.csv", length(top_features))
    y_train <- read.csv(y_train_file)$label
    
    model_lr <- glm(y_train ~ ., data = X_train_scaled, family = binomial)
    probs <- predict(model_lr, newdata = user_scaled, type = "response")
    

    labels <- ifelse(probs > 0.5, "Toxic", "Non-toxic")
    

    risk_factor <- mean(labels == "Toxic")
    
    output$results_preview <- renderTable({
      data.frame(Molecule = 1:nrow(user_data),
                 Probability_Toxic = probs,
                 Label = labels)
    })
    
    output$status <- renderText(
      paste0("User dataset processed.\nPredicted ", round(risk_factor*100,1),
             "% of molecules are toxic")
    )
  })
  
}

shinyApp(ui, server)
