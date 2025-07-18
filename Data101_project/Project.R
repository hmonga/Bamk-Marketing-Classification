library(readr)
library(dplyr)
library(ggplot2)
library(rpart)
library(rpart.plot)
library(e1071)
library(caret)

getwd()
setwd("/Users/harshmonga/Desktop/Data101_project")

bank_data <- read_delim("bank-full.csv", delim = ";")

str(bank_data)
summary(bank_data)

bank_data$y <- as.factor(bank_data$y) 
cat("Class Balance Table:\n")
table(bank_data$y)

cat("Class Balance Proportion:\n")
prop.table(table(bank_data$y))

ggplot(bank_data, aes(x = age, fill = y)) +
  geom_histogram(binwidth = 5, position = "dodge") +
  labs(title = "Age Distribution by Subscription", x = "Age", fill = "Subscribed")

ggplot(bank_data, aes(x = job, fill = y)) +
  geom_bar(position = "fill") +
  coord_flip() +
  labs(title = "Subscription Rate by Job", y = "Proportion", x = "Job")

ggplot(bank_data, aes(x = education, fill = y)) +
  geom_bar(position = "fill") +
  labs(title = "Subscription Rate by Education", y = "Proportion", x = "Education")

set.seed(42)
train_index <- createDataPartition(bank_data$y, p = 0.7, list = FALSE)
train_data <- bank_data[train_index, ]
test_data  <- bank_data[-train_index, ]

# Fit decision tree
tree_model <- rpart(y ~ ., data = train_data, method = "class")

# Plot tree
rpart.plot(tree_model)

# Predict on test set
tree_preds <- predict(tree_model, test_data, type = "class")

# Confusion Matrix
confusionMatrix(tree_preds, test_data$y)

# Fit the Naïve Bayes model
nb_model <- naiveBayes(y ~ ., data = train_data)

# Predict on the test set
nb_preds <- predict(nb_model, test_data)

# Confusion matrix for Naïve Bayes
confusionMatrix(nb_preds, test_data$y)

# Print accuracy of both models
tree_cm <- confusionMatrix(tree_preds, test_data$y)
cat("Decision Tree Accuracy:", tree_cm$overall["Accuracy"], "\n")

nb_cm <- confusionMatrix(nb_preds, test_data$y)
cat("Naïve Bayes Accuracy:", nb_cm$overall["Accuracy"], "\n")
