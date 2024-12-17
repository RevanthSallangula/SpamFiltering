# Load all required libraries
library(readr)
library(stringr)
library(dplyr)
library(purrr)
library(tidytext)
library(tidyr)

spam <- read_tsv("Datasets/spam.tsv", col_names = c("label", "sms"))

spec(spam)

n <- nrow(spam)
n_training <- floor(0.8 * n)  
n_cv <- floor(0.1 * n)
n_test <- n - n_training - n_cv  

set.seed(123)  
train_indices <- sample(1:n, size = n_training, replace = FALSE)

remaining_indices <- setdiff(1:n, train_indices)

cv_indices <- remaining_indices[1:n_cv]
test_indices <- remaining_indices[(n_cv + 1):length(remaining_indices)]

spam_train <- spam[train_indices, ]
spam_cv <- spam[cv_indices, ]
spam_test <- spam[test_indices, ]

print(mean(spam_train$label == "ham"))
print(mean(spam_cv$label == "ham"))
print(mean(spam_test$label == "ham"))


tidy_train <- spam_train %>% 
  mutate(
    sms = str_to_lower(sms) %>% 
      str_squish %>% 
      str_replace_all("[[:punct:]]", "") %>% 
      str_replace_all("[\u0094\u0092\u0096\n\t]", "") %>% 
      str_replace_all("[[:digit:]]", "")
  )

# Creating the vocabulary
vocabulary <- NULL
messages <- tidy_train %>%  pull(sms)

for (m in messages) {
  words <- str_split(m, " ")[[1]]
  vocabulary <- c(vocabulary, words)
}

vocabulary <- vocabulary %>% unique()

spam_messages <- tidy_train %>% 
  filter(label == "spam") %>% 
  pull(sms)

ham_messages <- tidy_train %>% 
  filter(label == "ham") %>% 
  pull(sms)

spam_vocab <- NULL
for (sm in spam_messages) {
  words <- str_split(sm, " ")[[1]]
  spam_vocab  <- c(spam_vocab, words)
}
spam_vocab

ham_vocab <- NULL
for (hm in ham_messages) {
  words <- str_split(hm, " ")[[1]]
  ham_vocab <- c(ham_vocab, words)
}
ham_vocab

n_spam <- spam_vocab %>% length()
n_ham <- ham_vocab %>% length()
n_vocabulary <- vocabulary %>% length()

p_spam <- mean(tidy_train$label == "spam")
p_ham <- mean(tidy_train$label == "ham")


spam_tokens <- tibble(text = spam_messages) %>%
  unnest_tokens(word, text) %>%
  count(word, name = "spam_count")

# Convert ham messages into a tidy format
ham_tokens <- tibble(text = ham_messages) %>%
  unnest_tokens(word, text) %>%
  count(word, name = "ham_count")

word_counts <- full_join(spam_tokens, ham_tokens, by = "word") %>%
  replace_na(list(spam_count = 0, ham_count = 0))

print(word_counts)

classify <- function(message, alpha = 1) {
  
  clean_message <- str_to_lower(message) %>% 
    str_squish %>% 
      str_replace_all("[[:punct:]]", "") %>% 
      str_replace_all("[\u0094\u0092\u0096\n\t]", "") %>% # Unicode characters
      str_replace_all("[[:digit:]]", "")
  
  words <- str_split(clean_message, " ")[[1]]
  

new_words <- setdiff(vocabulary, words)

new_word_probs <- tibble(
  word = new_words,
  spam_prob = 1,
  ham_prob = 1
)

present_probs <- word_counts %>% 
  filter(word %in% words) %>% 
  mutate(
    spam_prob = (spam_count + alpha) / (n_spam + alpha * n_vocabulary),
    ham_prob = (ham_count + alpha) / (n_ham + alpha * n_vocabulary)
  ) %>% 
  bind_rows(new_word_probs) %>% 
  pivot_longer(
    cols = c("spam_prob", "ham_prob"),
    names_to = "label",
    values_to = "prob"
  ) %>% 
  group_by(label) %>% 
  summarize(
    wi_prob = prod(prob) 
  )

p_spam_given_message <- p_spam * (present_probs %>% filter(label == "spam_prob") %>% pull(wi_prob))
p_ham_given_message <- p_ham * (present_probs %>% filter(label == "ham_prob") %>% pull(wi_prob))

ifelse(p_spam_given_message >= p_ham_given_message, "spam", "ham")
}

final_train <- tidy_train %>% 
  mutate(
    prediction = map_chr(sms, function(m) { classify(m) })
  ) 

confusion <- table(final_train$label, final_train$prediction)
accuracy <- (confusion[1,1] + confusion[2,2]) / nrow(final_train)

alpha_grid <- seq(0.05, 1, by = 0.05)
cv_accuracy <- NULL

for (alpha in alpha_grid) {
  
  cv_probs <- word_counts %>% 
    mutate(
      spam_prob = (spam_count + alpha / (n_spam + alpha * n_vocabulary)),
      ham_prob = (ham_count + alpha) / (n_ham + alpha * n_vocabulary)
    )
  
  cv <- spam_cv %>% 
    mutate(
      prediction = map_chr(sms, function(m) { classify(m, alpha = alpha) })
    ) 
  
  confusion <- table(cv$label, cv$prediction)
  acc <- (confusion[1,1] + confusion[2,2]) / nrow(cv)
  cv_accuracy <- c(cv_accuracy, acc)
}

tibble(
  alpha = alpha_grid,
  accuracy = cv_accuracy
)

optimal_alpha <- 0.1

spam_test <- spam_test %>% 
  mutate(
    prediction = map_chr(sms, function(m) { classify(m, alpha = optimal_alpha)} )
  )

confusion <- table(spam_test$label, spam_test$prediction)
test_accuracy <- (confusion[1,1] + confusion[2,2]) / nrow(spam_test)
test_accuracy