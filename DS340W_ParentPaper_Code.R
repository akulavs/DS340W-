library(readr)
library(dplyr)
library(stringr)
library(lubridate)
library(Matrix)
library(ranger)
library(glmnet)
library(caret)
library(yardstick)

set.seed(42)

# ----------------------------
# 0) Load
# ----------------------------
CSV_PATH <- "C:/Users/steve/Downloads/facebook-fact-check.csv"
df <- read_csv(CSV_PATH, show_col_types = FALSE)

# ----------------------------
# 1) Diagnostics: non-empty counts + unique ratings
# ----------------------------
nonempty_counts <- sort(
  sapply(df, function(x) sum(!is.na(x) & trimws(as.character(x)) != "")),
  decreasing = TRUE
)
cat("\n=== Top 15 columns by non-empty cell count ===\n")
print(head(nonempty_counts, 15))

rating_col <- if ("Rating" %in% names(df)) "Rating" else if ("rating" %in% names(df)) "rating" else NA
if (is.na(rating_col)) stop("Could not find a Rating column (expected 'Rating' or 'rating').")

cat("\n=== Unique Rating values (raw) ===\n")
print(sort(unique(tolower(trimws(as.character(df[[rating_col]]))))))

# ----------------------------
# 2) Standardize column names
# ----------------------------
df <- df %>% rename_with(~ str_replace_all(tolower(.x), "\\s+", "_"))
if (!"rating" %in% names(df)) stop("Expected a 'rating' column after renaming.")

# ----------------------------
# 3) Build label y from rating (1=fake, 0=real)
#    NOTE: mapping is generic; if you end up with only one class, edit tokens.
# ----------------------------
make_label_from_rating <- function(r) {
  r <- tolower(trimws(as.character(r)))
  r[is.na(r)] <- ""
  
  fake_tokens <- c(
    "false", "pants on fire", "incorrect", "misleading", "mostly false",
    "fake", "hoax", "scam", "no evidence", "unproven", "unsupported"
  )
  real_tokens <- c(
    "true", "mostly true", "correct", "accurate", "real"
  )
  
  y <- ifelse(r %in% fake_tokens, 1L,
              ifelse(r %in% real_tokens, 0L, NA_integer_))
  
  # heuristic fallback for unmapped
  miss <- is.na(y)
  if (any(miss)) {
    y[miss] <- ifelse(str_detect(r[miss], "false|fake|pants on fire|mislead|hoax|scam|incorrect"),
                      1L,
                      ifelse(str_detect(r[miss], "\\btrue\\b|accurate|correct"),
                             0L, NA_integer_))
  }
  y
}

df$y <- make_label_from_rating(df$rating)

cat("\n=== Label mapping summary (y: 1=fake, 0=real) ===\n")
print(table(df$y, useNA = "ifany"))

# keep labeled rows only
df_model <- df %>% filter(!is.na(y))
if (nrow(df_model) < 50) warning("Very few labeled rows after mapping; you may need to adjust rating->label mapping.")

# ----------------------------
# 4) Metadata-based transformations (since title/text are empty)
# ----------------------------
normalize_url <- function(u) {
  u <- trimws(as.character(u))
  u[is.na(u)] <- ""
  missing_scheme <- !str_detect(u, "^https?://") & u != ""
  u[missing_scheme] <- paste0("http://", u[missing_scheme])
  
  domain <- str_replace(u, "^https?://", "")
  domain <- str_replace(domain, "/.*$", "")
  domain <- tolower(str_replace(domain, "^www\\.", ""))
  
  path <- str_replace(u, "^https?://[^/]+", "")
  path <- str_replace(path, "/+$", "")
  
  str_trim(paste0(domain, path))
}

# ensure required columns exist
if (!"post_url" %in% names(df_model)) df_model$post_url <- ""
if (!"page" %in% names(df_model)) df_model$page <- ""
if (!"date_published" %in% names(df_model)) df_model$date_published <- NA_character_
if (!"post_type" %in% names(df_model)) df_model$post_type <- ""
if (!"category" %in% names(df_model)) df_model$category <- ""
if (!"debate" %in% names(df_model)) df_model$debate <- ""
if (!"share_count" %in% names(df_model)) df_model$share_count <- 0
if (!"reaction_count" %in% names(df_model)) df_model$reaction_count <- 0
if (!"comment_count" %in% names(df_model)) df_model$comment_count <- 0

df_model <- df_model %>%
  mutate(
    post_url_norm = normalize_url(post_url),
    page_norm     = normalize_url(page),
    
    # date parsing (tries datetime then date)
    date_published_parsed = suppressWarnings(ymd_hms(date_published, tz = "UTC")),
    date_published_parsed = ifelse(is.na(date_published_parsed),
                                   suppressWarnings(ymd(date_published)),
                                   date_published_parsed),
    date_published_parsed = as.POSIXct(date_published_parsed, origin="1970-01-01", tz="UTC"),
    
    # numeric coercion + NA->0
    share_count    = suppressWarnings(as.numeric(share_count)),
    reaction_count = suppressWarnings(as.numeric(reaction_count)),
    comment_count  = suppressWarnings(as.numeric(comment_count)),
    share_count    = ifelse(is.na(share_count), 0, share_count),
    reaction_count = ifelse(is.na(reaction_count), 0, reaction_count),
    comment_count  = ifelse(is.na(comment_count), 0, comment_count),
    
    # clean categoricals
    category  = str_to_lower(trimws(as.character(category))),
    post_type = str_to_lower(trimws(as.character(post_type))),
    debate    = str_to_lower(trimws(as.character(debate))),
    
    # engineered
    has_debate = ifelse(debate == "" | is.na(debate) | debate %in% c("no","false","0"), 0L, 1L),
    log_shares = log1p(share_count),
    log_reactions = log1p(reaction_count),
    log_comments = log1p(comment_count),
    weekday = ifelse(is.na(date_published_parsed), NA_character_, as.character(wday(date_published_parsed, label = TRUE, abbr = TRUE))),
    hour    = ifelse(is.na(date_published_parsed), NA_integer_, hour(date_published_parsed))
  )

# ----------------------------
# 5) Build feature frame + DROP 1-level predictors (fix for contrasts error)
# ----------------------------
feature_df <- df_model %>%
  transmute(
    y = factor(y, levels = c(0, 1)),   # 0=real, 1=fake
    share_count, reaction_count, comment_count,
    log_shares, log_reactions, log_comments,
    has_debate,
    category  = ifelse(is.na(category)  | category  == "", "unknown", category),
    post_type = ifelse(is.na(post_type) | post_type == "", "unknown", post_type),
    weekday   = ifelse(is.na(weekday)   | weekday   == "", "unknown", weekday),
    hour
  ) %>%
  mutate(
    # numeric NA handling
    hour = ifelse(is.na(hour), round(median(hour, na.rm = TRUE)), hour),
    
    # convert categoricals to factors
    category  = factor(category),
    post_type = factor(post_type),
    weekday   = factor(weekday)
  )

# Ensure y has 2+ classes (otherwise modeling is impossible)
if (length(unique(feature_df$y)) < 2) {
  stop("Target y has < 2 classes after rating->label mapping. Adjust make_label_from_rating() to match your Rating values.")
}

# Drop predictors with only 1 unique value/level
predictor_cols <- setdiff(names(feature_df), "y")
one_level <- predictor_cols[sapply(feature_df[predictor_cols], function(col) length(unique(col)) < 2)]

if (length(one_level) > 0) {
  cat("\nDropping 1-level predictors (prevents contrasts error):\n")
  print(one_level)
  feature_df <- feature_df %>% select(-all_of(one_level))
}

# Sparse one-hot encoding
X <- sparse.model.matrix(y ~ ., data = feature_df)[, -1, drop = FALSE]
y_fac <- feature_df$y
y_num <- as.integer(as.character(y_fac))  # 0/1 for glmnet

cat("\n=== Final feature matrix ===\n")
cat("Rows:", nrow(X), "Cols:", ncol(X), "\n")
cat("Class balance:\n")
print(table(y_fac))

# ----------------------------
# 6) Train/test split (80/20 stratified)
# ----------------------------
idx <- createDataPartition(y_fac, p = 0.8, list = FALSE)
X_train <- X[idx, ]
X_test  <- X[-idx, ]
y_train <- y_fac[idx]
y_test  <- y_fac[-idx]
y_train_num <- y_num[idx]

# ----------------------------
# 7) Random Forest (500 trees) + OOB error
# ----------------------------
cat("\n=== Random Forest (500 trees) ===\n")
rf_model <- ranger(
  x = X_train,
  y = y_train,
  num.trees = 500,
  probability = TRUE,
  oob.error = TRUE,
  seed = 42
)
rf_oob_error <- rf_model$prediction.error
cat("OOB error rate:", round(rf_oob_error * 100, 2), "%\n")

rf_prob <- predict(rf_model, data = X_test)$predictions[, "1"]
rf_pred <- factor(ifelse(rf_prob >= 0.5, 1, 0), levels = c(0, 1))

# ----------------------------
# 8) Logistic Regression (glmnet Elastic Net)
# ----------------------------
cat("\n=== Logistic Regression (Elastic Net) ===\n")
alpha_val <- 0.5

cv_lr <- cv.glmnet(
  x = X_train,
  y = y_train_num,
  family = "binomial",
  alpha = alpha_val,
  nfolds = 5
)

lr_model <- glmnet(
  x = X_train,
  y = y_train_num,
  family = "binomial",
  alpha = alpha_val,
  lambda = cv_lr$lambda.min
)

lr_prob <- as.numeric(predict(lr_model, newx = X_test, type = "response"))
lr_pred <- factor(ifelse(lr_prob >= 0.5, 1, 0), levels = c(0, 1))

# ----------------------------
# 9) Evaluation: accuracy, precision, recall, f1 + confusion matrices
# ----------------------------
eval_metrics <- function(truth, pred) {
  tibble(
    accuracy  = accuracy_vec(truth, pred),
    precision = precision_vec(truth, pred, event_level = "second"),
    recall    = recall_vec(truth, pred, event_level = "second"),
    f1        = f_meas_vec(truth, pred, event_level = "second")
  )
}

rf_metrics <- eval_metrics(y_test, rf_pred) %>% mutate(model = "RandomForest")
lr_metrics <- eval_metrics(y_test, lr_pred) %>% mutate(model = "LogReg_ElasticNet")

results <- bind_rows(rf_metrics, lr_metrics) %>%
  select(model, everything()) %>%
  arrange(desc(f1))

cat("\n=== Test-set Metrics (threshold=0.5) ===\n")
print(results)

cat("\n--- Random Forest Confusion Matrix ---\n")
print(confusionMatrix(rf_pred, y_test))

cat("\n--- Logistic Regression Confusion Matrix ---\n")
print(confusionMatrix(lr_pred, y_test))
