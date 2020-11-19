# Code adapted from https://tensorflow.rstudio.com/tutorials/advanced/images/cnn/

# Load benchmarking library -----------------------------------------------
library(tictoc)

# Config the plaidml backend -------------------------------------------
library(keras)
use_condaenv("r-plaidml-keras", required = TRUE) # remember to rename to your own environment with plaidml
use_backend("plaidml")
use_backend("plaidml") # sometimes plaidml requires this line to be called twice to get it working.


# Load dataset -----------------------------------------------------

cifar <- dataset_cifar10()

class_names <- c('airplane', 'automobile', 'bird', 'cat', 'deer',
                 'dog', 'frog', 'horse', 'ship', 'truck')

# index <- 1:30

# par(mfcol = c(5,6), mar = rep(1, 4), oma = rep(0.2, 4))
# cifar$train$x[index,,,] %>% 
#   purrr::array_tree(1) %>%
#   purrr::set_names(class_names[cifar$train$y[index] + 1]) %>% 
#   purrr::map(as.raster, max = 255) %>%
#   purrr::iwalk(~{plot(.x); title(.y)})


# Instantiate model -------------------------------------------------------

model <- keras_model_sequential() %>% 
  layer_conv_2d(filters = 32, kernel_size = c(3,3), activation = "relu", 
                input_shape = c(32,32,3)) %>% 
  layer_max_pooling_2d(pool_size = c(2,2)) %>% 
  layer_conv_2d(filters = 64, kernel_size = c(3,3), activation = "relu") %>% 
  layer_max_pooling_2d(pool_size = c(2,2)) %>% 
  layer_conv_2d(filters = 64, kernel_size = c(3,3), activation = "relu")

summary(model)

model %>% 
  layer_flatten() %>% 
  layer_dense(units = 64, activation = "relu") %>% 
  layer_dense(units = 10, activation = "softmax")

model %>% compile(
  optimizer = "adam",
  loss = "sparse_categorical_crossentropy",
  metrics = "accuracy"
)


# Train model - THIS IS THE TIMED PART -------------------------------------------------------------
tic("Training a simple CNN model with plaidml using the GPU")
history <- model %>% 
  fit(
    x = cifar$train$x, y = cifar$train$y,
    epochs = 10,
    validation_data = unname(cifar$test),
    verbose = 2
  )
toc()
