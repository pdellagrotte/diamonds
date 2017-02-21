###Assignment 2### 
# Author: Paul DellaGrotte


library(ggplot2) # Charts
library(pastecs) # descriptive statistics
library(rpart) # decision tree
library(rpart.plot)
library(randomForest)
library(MASS)
library(reshape2)
library(lattice)
library(leaps)
library(glmnet)
library(plyr)

######################################################
############### Load Data ############################
######################################################

data <- "filepath of csv"

# read in data from web, no header
df <- read.csv(data, header=TRUE)

df2 <- df # create data frame for dummy variables

######################################################

######################################################
############### Data Quality Check ###################
######################################################

head(df)
names(df)
stat.desc(df) # print table of descriptive stats

######################################################

######################################################
############## Data Transformations ##################
######################################################

# Create dummy variables for each categorical variable

for(level in unique(df2$color)){
  df2[paste("color", level, sep = "_")] <- ifelse(df2$color == level, 1, 0)
}

for(level in unique(df2$clarity)){
  df2[paste("clarity", level, sep = "_")] <- ifelse(df2$clarity == level, 1, 0)
}

for(level in unique(df2$store)){
  df2[paste("store", level, sep = "_")] <- ifelse(df2$store == level, 1, 0)
}

for(level in unique(df2$channel)){
  df2[paste("channel", level, sep = "_")] <- ifelse(df2$channel == level, 1, 0)
}

#Remove redundant x variables
df2$color <- NULL
df2$clarity <- NULL
df2$channel <- NULL
df2$store <- NULL



#Remove reference category (e.g. color_1, clarity_2, ect)
df2$color_1 <- NULL
df2$clarity_2 <- NULL
df2$store_Ashford <- NULL
df2$channel_Independent <-NULL

#Remove spaces from names to be read as continuous string
names(df2)[names(df2)=="store_R. Holland"] <- "store_RHolland"
names(df2)[names(df2)=="store_Fred Meyer"] <- "store_FredMeyer"
names(df2)[names(df2)=="store_Blue Nile"] <- "store_BlueNile"

#Add log transformations of price and carat
df2$logprice <- log(df2$price)
df2$logcarat<- log(df2$carat)
df2$price <- NULL
df2$carat <-NULL


# Print results of transformations
View(df2) # check to make sure df2 has all proper dummy variables
str(df2) # Show structure

######################################################

######################################################
####################### EDA ##########################
######################################################

hist(df$price)
hist(df$carat)

scale_x <- scale_x_continuous(limits = c(0, 3),
                              breaks = round(seq(0, max(df$carat), by = .25),2))

scale_y <- scale_y_continuous(limits = c(0, 30000),
                              labels = scales::dollar, 
                              breaks = round(seq(0, max(df$price), by = 5000),2))

gcorr<- round(cor(df$carat, df$price),4) # Correlation for display


ggplot(df, aes(x=carat, y=price, color=color, shape=cut)) + 
  geom_point() +
  scale_y + scale_x + 
  labs(title=paste("Correlation=",gcorr), x = "carat", y= "price") +
  theme(plot.title = element_text(face="bold", size=rel(1.25)))

ggplot(df, aes(carat, price)) + geom_point() + geom_smooth() + 
  labs(x="carat", y="price") + scale_x + scale_y

ggplot(df, aes(log(carat), log(price))) + geom_point() + geom_smooth()+
  labs(x="log(carat)", y="log(price)")

gplot1 <- ggplot(data = df, aes(color, price)) + theme(legend.position="none")
gplot1 + geom_boxplot(aes(fill = color)) + scale_y

gplot2 <- ggplot(data = df, aes(channel, price)) 
gplot2 + geom_boxplot(aes(fill = channel)) + scale_y + theme(legend.position="none")

gplot2 <- ggplot(data = df, aes(cut, price)) 
gplot2 + geom_boxplot(aes(fill = cut)) + scale_y + theme(legend.position="none")

gplot2 <- ggplot(data = df, aes(clarity, price))
gplot2 + geom_boxplot(aes(fill = clarity)) + scale_y + theme(legend.position="none")

gplot3 <- ggplot(data = df, aes(store, price))
gplot3 + geom_boxplot(aes(fill = cut)) + scale_y

gplot3 <- ggplot(data = df, aes(clarity, price))
gplot3 + geom_boxplot(aes(fill = cut))

gplot4 <- ggplot(data = df, aes(carat, price))
gplot4 + geom_point(color="red")
gplot4 + geom_point(aes(color=cut))

ggplot(df, aes(x=carat, y=price, color=clarity)) + 
  geom_point() + facet_grid(~ cut)

gplot5 <- ggplot(df, aes(color, fill=cut)) + geom_bar()

ggplot(df, aes(price, color=cut)) + geom_freqpoly(binwidth=1000)

# looks like ideal cut is bimodal for price & carat
ggplot(df, aes(price, fill=cut)) + geom_histogram(alpha = 0.5, binwidth =600)
ggplot(df, aes(carat, fill=cut)) + geom_histogram(binwidth =0.4)

hist(df$price, freq = F, main=" ", xlab= "Price")
curve(dnorm(x, mean=mean(df$price),sd=sd(df$price)), add = T, col="red", lwd=2)

hist(df$carat, freq = F, main=" ", xlab= "Carat")
curve(dnorm(x, mean=mean(df$carat),sd=sd(df$carat)), add=T, col="red", lwd=2)


### Decision Tree for EDA #####
M0 <- rpart(price ~ ., data=df, method="anova")
summary(M0)

rpart.plot(M0) # plot model

###############################

######################################################

######################################################
############## Split Training-Testing ################
######################################################

# 70 / 30 Split per assignment instructions

set.seed(1200) # set the seed so randomness is reproducable
g <- runif(nrow(df2)) # set a bunch of random numbers as rows
df_random <- df2[order(g),] # reorder the data set

train_size <- floor(.70 * nrow(df2))  # Select % of data set to use for training
test_size <- nrow(df2) - train_size  # use remainder of data set for testing

df_train <- df_random[1:train_size,]
df_test <- df_random[(train_size+1):nrow(df2),]

######################################################

######################################################
####################### Models ########################
######################################################

# Functions to compute R-Squared and RMSE
rsq <- function(y,f) {1 - sum((y-f)^2)/sum((y-mean(y))^2) } 
rmse <- function(y, f) {sqrt(mean((y-f)^2)) }

### Decision Tree #####
M0 <- rpart(logprice ~ ., data=df_train, method="anova")

p0 <- predict(M0, newdata=df_test)   #set type = to class to get correct output
plot(df_test$logprice, p0)

actual <- df_test$logprice
predicted <- p0

rsq(actual,predicted)
rmse(actual,predicted)

# On Training

p0 <- predict(M0, newdata=df_train)   #set type = to class to get correct output

actual <- df_train$logprice
predicted <- p0

rsq(actual,predicted)
rmse(actual,predicted)

########################

#### Single Variable ###
M1<- lm(logprice ~ logcarat, data=df_train)

p1 <- predict(M1, newdata=df_test) 

actual <- df_test$logprice
predicted <- p1
error <- actual - predicted

rsq(actual,predicted)
rmse(actual,predicted)

# On Training
p1 <- predict(M1, newdata=df_train) 

actual <- df_train$logprice
predicted <- p1
error <- actual - predicted

rsq(actual,predicted)
rmse(actual,predicted)

summary(M1)$r.squared


########################

########################

## Variable Selection ##

M2 <- lm(logprice~ ., data = df_train)

step_b <- step(M2, direction = "backward")
step_f <- step(M2, direction = "forward")
step_s <- step(M2, direction = "both")

listRsqu <- list()

c(listRsqu, a=summary(step_b)$r.squared, b=summary(step_f)$r.squared, c=summary(step_s)$r.squared)


listRsqu # best is forward selection

p4 <- predict(step_f, newdata=df_test)

actual <- df_test$logprice
predicted <- p4
error <- actual - predicted

rsq(actual,predicted)
rmse(actual,predicted)


# On Training
p4 <- predict(step_f, newdata=df_train)

actual <- df_train$logprice
predicted <- p4
error <- actual - predicted

rsq(actual,predicted)
rmse(actual,predicted)

########################

## Model w/ Interaction ##

M3 <- lm(logprice~ logcarat+cut*channel_Internet, data = df_train)
summary(M3)$r.squared

M3 <-lm(formula = logprice ~ cut + color_4 + color_5 + color_7 + color_8 + 
     color_3 + color_2 + color_6 + color_9 + clarity_7 + clarity_6 + 
     clarity_4 + clarity_8 + clarity_9 + clarity_5 + clarity_10 + 
     clarity_3 + store_Goodmans + store_Chalmers + store_FredMeyer + 
     store_RHolland + store_Ausmans + store_University + store_Kay + 
     store_Zales + store_Danford + store_BlueNile + store_Riddles + 
     channel_Mall + channel_Internet + logcarat + channel_Internet*cut, data = df_train)

p4 <- predict(M3, newdata=df_test)

actual <- df_test$logprice
predicted <- p4
error <- actual - predicted

rsq(actual,predicted)
rmse(actual,predicted)

# On Training

p4 <- predict(M3, newdata=df_train)

actual <- df_train$logprice
predicted <- p4
error <- actual - predicted

rsq(actual,predicted)
rmse(actual,predicted)


########################


####### LASSO ##########

xfactors <- model.matrix(df$price ~ df$carat + 
                           df$color + df$clarity + 
                           df$cut + df$channel + df$store)


xfactors <-model.matrix(data = df2,logprice ~ cut + color_4 + color_5 + color_7 + color_8 + 
          color_3 + color_2 + color_6 + color_9 + clarity_7 + clarity_6 + 
          clarity_4 + clarity_8 + clarity_9 + clarity_5 + clarity_10 + 
          clarity_3 + store_Goodmans + store_Chalmers + store_FredMeyer + 
          store_RHolland + store_Ausmans + store_University + store_Kay + 
          store_Zales + store_Danford + store_BlueNile + store_Riddles + 
          channel_Mall + channel_Internet + logcarat)


fit = glmnet(xfactors, y = df$price, alpha = 1)

plot(fit)
coef(fit)

summary(fit)


########################

#### Random Forest #####

M4 <-randomForest(logprice ~ ., data=df_train, replace=T,ntree=100)

#vars<-dimnames(imp)[[1]]
#imp<- data.frame(vars=vars, imp=as.numeric(imp[,1]))
#imp<-imp[order(imp$imp,decreasing=T),]

par(mfrow=c(1,2))
varImpPlot(M4, main="Variable Importance Plot: Base Model")
plot(M4, main="Error vs. No. of Trees Plot: Base Model")

p4<- predict(object=M4, newdata = df_test)

actual <- df_test$logprice
predicted <- p4
error <- actual - predicted

rsq(actual,predicted)
rmse(actual,predicted)

#On Training
p4<- predict(object=M4, newdata = df_train)

actual <- df_train$logprice
predicted <- p4
error <- actual - predicted

rsq(actual,predicted)
rmse(actual,predicted)

########################

######################################################

######################################################
############### Model Comparison #####################
######################################################
#coefficients(m1) # model coefficients
#confint(m1, level=0.95) # CIs for model parameters 
#m1ted(m1) # predicted values
#residuals(m1) # residuals
#anova(m1) # anova table 
#vcov(m1) # covariance matrix for model parameters 
#influence(m1) # regression diagnostics
######################################################