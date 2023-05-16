library(ricu)
library(ggplot2)
library(ranger)
library(data.table)
library(faircause)
library(xgboost)
library(latex2exp)

src <- "miiv"
dat <- load_concepts(c("vent_ind", "resp", "po2", "sofa", "o2sat", "sex", "age"), 
                     src)

dat <- dat[get(index_var(dat)) <= hours(48L) & get(index_var(dat)) >= hours(0L)]
dat[is.na(vent_ind), vent_ind := FALSE]

dat[, is_vent := cummax(vent_ind), by = c(id_vars(dat))]

# cohort generation
cand <- unique(id_col(dat[o2sat <= 96 & is_vent == 0]))
cdat <- dat[id_col(dat) %in% cand]
cdat <- replace_na(cdat, type = "locf", vars = c("o2sat", "po2", "resp"))
cdat[is.na(po2), po2 := median(po2, na.rm = TRUE)]
cdat[is.na(resp), resp := median(resp, na.rm = TRUE)]

# lag by 3 hours both ways
cdat[, is_vent_lag3 := data.table::shift(is_vent, -3L)]
cdat[, is_vent_lagrev3 := data.table::shift(is_vent, 3L)]

# the actioned cohort
act <- merge(
  cdat[is_vent == 0 & is_vent_lag3 == 1, 
       list(o2prior = mean(o2sat, na.rm = TRUE), sofa = max(sofa),
            resp = mean(resp, na.rm = TRUE), po2 = mean(po2, na.rm = TRUE),
            sex = unique(sex), age = unique(age)), 
       by = c(id_vars(dat))],
  cdat[is_vent == 1 & is_vent_lagrev3 == 0, 
       list(o2post = mean(o2sat, na.rm = TRUE)),
       by = c(id_vars(dat))]
)

act[, respirator := 1]

# take complete cases
act <- act[complete.cases(act)]

# the non-actioned cohort
ctrls <- id_col(cdat[, max(is_vent), by = "stay_id"][V1 == 0])
ndat <- cdat[(id_col(cdat) %in% ctrls)]

#' * should this be a random selection? *
skp <- merge(
  ndat[get(index_var(ndat)) %in% hours(10, 11, 12), 
       list(o2prior = mean(o2sat, na.rm = TRUE), sofa = max(sofa),
            resp = mean(resp, na.rm = TRUE), po2 = mean(po2, na.rm = TRUE),
            sex = unique(sex), age = unique(age)), 
       by = c(id_vars(dat))],
  ndat[get(index_var(ndat)) %in% hours(13, 14, 15), 
       list(o2post = mean(o2sat, na.rm = TRUE)), 
       by = c(id_vars(dat))]
)
skp <- skp[, respirator  := 0]
skp <- skp[complete.cases(skp)]

res <- rbind(act, skp)
#' * data construction ends *

#' * start modeling *
set.seed(2023)

# create the SFM
X <- "sex"
Z <- "age"
W <- c("sofa", "po2", "resp", "o2prior")
D <- "respirator"
Y <- "o2post"

# compute P(d | x1) - P(d | x0) for current policy
res[, list(pd_x = mean(respirator)), by = "sex"]

# make sex 0/1
res[, sex := ifelse(sex == "Male", 1, 0)]

# decompose the original policy
fcb_org <- fairness_cookbook(
  res[o2prior < 97], X = X, Z = Z, W = W,
  Y = "respirator", x0 = 0, x1 = 1
)
summary(fcb_org)
autoplot(fcb_org, signed = FALSE) + ggtitle(NULL) +
  scale_x_discrete(labels = c("TV", "DE", "IE", "SE"))
ggsave("d-curr-decomposed.png", width = 6, height = 4)

# set Y as the post intervention o2
res[, y := o2post]

# estimate the benefit (could use xgboost instead to check...)
forest <- FALSE
if (forest) {
  
  rf <- ranger(y ~ o2prior + sofa + sex + age + respirator, data = res)
  
  res0 <- res1 <- copy(res)
  res0[, respirator := 0]
  y0 <- predict(rf, res0)$predictions
  
  res1[, respirator := 1]
  y1 <- predict(rf, res1)$predictions
} else {

  xgbcv <- xgb.cv(params = list(eta = 0.1), 
                  data = as.matrix(res[, c(X, Z, W, D), with = FALSE]), 
                  label = res[[Y]], nrounds = 100, early_stopping_rounds = 3, 
                  nfold = 10)
  
  # pick optimal number of rounds
  nrounds <- xgbcv$best_iteration
  
  # train the prediction object
  xgb <- xgboost(params = list(eta = 0.1), 
                 data = as.matrix(res[, c(X, Z, W, D), with = FALSE]), 
                 label = res[[Y]], nrounds = nrounds, )
  
  res0 <- res1 <- copy(res)
  res0[[D]] <- 0
  res1[[D]] <- 1
  
  y0 <- predict(xgb, as.matrix(res0[, c(X, Z, W, D), with = FALSE]))
  y1 <- predict(xgb, as.matrix(res1[, c(X, Z, W, D), with = FALSE]))
}

res[, y1 := y1]
res[, y0 := y0]
adjust <- TRUE
if (adjust) res[y1 < y0, y1 := y0] else res <- res[y1 > y0]
# enforce monotonicity of the treatment
res[, delta := y1 - y0]

# o2sat loss function
o2sat_loss <- function(x) ifelse(x < 97, -(x-97)^2, 0)
# compute f(benefit)
res[, fdelta := o2sat_loss(y1) - o2sat_loss(y0)]

# group into deciles
res[, dec := .bincode(fdelta, quantile(fdelta, seq(0, 1, 0.1)), 
                                       include.lowest = TRUE)]

ggplot(res[, mean(respirator), by = c("sex", "dec")],
       aes(x = dec, y = V1, color = factor(sex))) + 
  geom_line() + geom_point() + theme_bw() +
  scale_x_continuous(breaks = 1:10, labels = paste0("D", 1:10)) +
  ylab("P(mechanical ventilation)") + xlab("Benefit Decile") +
  scale_color_discrete(name = "Sex", labels = c("Female", "Male")) +
  theme(legend.position = c(0.2, 0.7), legend.box.background = element_rect())

ggsave("benefit-calibration.png", height = 4 * 0.8, width = 6 * 0.8)

# construct a policy using Alg. 1
budget <- mean(res$respirator)
interior <- floor(100 * budget)
boundary <- interior + 1

# treat the interior
res[fdelta >= quantile(fdelta, 1 - interior / 100), respirator_opt := 1]
int_tot <- sum(res$respirator_opt, na.rm = TRUE)

# boundary
bnd_tot <- sum(res$respirator) - int_tot

# treat males on boundary
bound <- res[fdelta <= quantile(fdelta, 1 - interior / 100) & 
               fdelta >= quantile(fdelta, 1 - boundary / 100)]

males <- sample(id_col(bound[sex == 1]), 
                size = sum(bound$sex) * bnd_tot / nrow(bound),
                replace = FALSE)
 
res[id_col(res) %in% males, respirator_opt := 1]

# treat females on boundary
females <- sample(id_col(bound[sex == 0]), 
                  size = sum(bound$sex == 0) * bnd_tot / nrow(bound),
                  replace = FALSE)

res[id_col(res) %in% females, respirator_opt := 1]

# remaining not treated
res[is.na(respirator_opt), respirator_opt := 0]

# apply Algorithm 2
res[, list(pd_x = mean(respirator_opt)), by = "sex"]

# decompose the gap
fcb_adj <- fairness_cookbook(res, X = X, Z = Z, W = W,
                             Y = "respirator_opt", x0 = 0, x1 = 1)
summary(fcb_adj)
autoplot(fcb_adj, signed = FALSE) + ggtitle(NULL) +
  scale_x_discrete(labels = c("TV", "DE", "IE", "SE"))
ggsave("d-star-decomposed.png", width = 6, height = 4)

# remove the extreme values which may influence the mean
extrm_val <- quantile(res$fdelta, 0.995)
# decompose the f-benefit
fcb_fdel <- fairness_cookbook(res[fdelta <= extrm_val], X = X, Z = Z, 
                              W = W, Y = "fdelta", 
                              x0 = 0, x1 = 1)

autoplot(fcb_fdel, signed = FALSE) + ggtitle(NULL) + 
  scale_x_discrete(labels = c("TV", "DE", "IE", "SE"))
ggsave("delta-decomposed.png", width = 6, height = 4)

# apply Algorithm 3
fairadapt <- TRUE
if (fairadapt) {
  # construct the diagram
  Ys <- c("y1", "y0")
  adj <- array(0, dim = c(length(c(X, Z, W, Ys)), length(c(X, Z, W, Ys))))
  colnames(adj) <- rownames(adj) <- c(X, Z, W, Ys)
  
  adj[X, c(W, Ys)] <- adj[Z, c(W, Ys)] <- adj[W, Ys] <- 1L
  adj[c("sofa", "resp", "po2"), "o2prior"] <- 
    adj[c("resp", "po2"), "sofa"] <- 1L
  
  # compute counterfactual values of Y_{d_1}, Y_{d_0}
  res[, sex := factor(sex, levels = c(1, 0))]
  
  # Y_{d_1} adjusted
  Y <- "y1"
  vars <- c(X, Z, W, Y)
  yd1_c <- fairadapt::fairadapt(
    as.formula(paste(Y, "~", paste(c(X, Z, W), collapse = "+"))),
    prot.attr = X, adj.mat = adj[vars, vars], train.data = res[, vars, with=FALSE]
  )
  res[, y1_c := fairadapt::adaptedData(yd1_c)$y1]
  
  # Y_{d_0} adjusted
  Y <- "y0"
  vars <- c(X, Z, W, Y)
  yd0_c <- fairadapt::fairadapt(
    as.formula(paste(Y, "~", paste(c(X, Z, W), collapse = "+"))),
    prot.attr = X, adj.mat = adj[vars, vars], train.data = res[, vars, with=FALSE]
  )
  res[, y0_c := fairadapt::adaptedData(yd0_c)$y0]
} else {
  
  res0[, sex := 1]
  res1[, sex := 1]
  
  y0_c <- predict(xgb, as.matrix(res0[, c(X, Z, W, D), with = FALSE]))
  y1_c <- predict(xgb, as.matrix(res1[, c(X, Z, W, D), with = FALSE]))
  res[, y0_c := y0_c]
  res[, y1_c := y1_c]
}

# compute counterfactual values of benefit
res[y1_c < y0_c, y1_c := y0_c]
res[, fdelta_c := o2sat_loss(y1_c) - o2sat_loss(y0_c)]

# give a causally-fair policy
act_tot <- sum(res$respirator)
delta_bc <- sort(res$fdelta_c, decreasing = TRUE)[act_tot]
res[, respirator_cf := fdelta_c >= delta_bc]

# check the disparity
fcb_cf <- fairness_cookbook(res, X = X, Z = Z, W = W, Y = "respirator_cf", 
                            x0 = 0, x1 = 1)

# combine the plots
# side-by-side plot of three decompositions
df <- rbind(
  cbind(summary(fcb_org)$measures, outcome = "curr"),
  cbind(summary(fcb_adj)$measures, outcome = "opt"),
  cbind(summary(fcb_cf)$measures, outcome = "cf")
)

df <- df[df$measure %in% c("ctfde", "ctfie", "ctfse", "tv"), ]
df[df$measure %in% c("ctfie", "ctfse"),]$value <- 
  -df[df$measure %in% c("ctfie", "ctfse"),]$value
ggplot(
  df,
  aes(x = factor(measure, levels = c("tv", "ctfde", "ctfie", "ctfse")),
      y = value,
      fill = factor(outcome, levels = c("curr", "opt", "cf")))
) +
  geom_bar(position=position_dodge(), stat="identity", colour='black') +
  geom_errorbar(aes(ymin = value - 1.96 * sd, ymax = value + 1.96 * sd),
                width=.2, position=position_dodge(.9)) +
  scale_fill_discrete(name = "Policy",
                      labels = c(TeX("$D^{curr}$"), TeX("$D^*$"),
                                 TeX("$D^{CF}$"))) +
  theme_bw() +
  theme(
    legend.position = "bottom",
    axis.text.x = element_text(size = 14),
    axis.title =  element_text(size = 16),
    legend.text = element_text(size = 14),
    legend.title = element_text(size = 14),
    title = element_text(size = 16)
  ) +
  xlab("Fairness Measure") + ylab("Value") +
  scale_x_discrete(labels = c(TeX("TV"),
                              TeX("DE"),
                              TeX("IE"),
                              TeX("SE"))) +
  scale_y_continuous(labels = scales::percent)

ggsave("policy-comparison.png", width = 6, height = 4)