##contextandspeaker TURK Analysis
# Elise Sugarman December 2013
# Bar graphs
rm(list=ls())

#--LOAD PACKAGES--
library(ggplot2)
library(boot)
library(bootstrap)
library(reshape2)

# LOAD FUNCTIONS
# code for bootstrapping 95% confidence intervals
theta <- function(x,xdata,na.rm=T) {mean(xdata[x],na.rm=na.rm)}
ci.low <- function(x,na.rm=T) {
  mean(x,na.rm=na.rm) - quantile(bootstrap(1:length(x),1000,theta,x,na.rm=na.rm)$thetastar,.025,na.rm=na.rm)}
ci.high <- function(x,na.rm=T) {
  quantile(bootstrap(1:length(x),1000,theta,x,na.rm=na.rm)$thetastar,.975,na.rm=na.rm) - mean(x,na.rm=na.rm)}

#--READ IN DATA--
setwd("/Documents/GRADUATE_SCHOOL/Projects/pragLearn/Elise Project/data/") # set working directory and read in data 
d <- read.csv("flowerdata_TURK.csv",header=TRUE)
mdata <- melt(d, id=c("example","flowerName", "condition" , "speaker", "context" , "generalizations"))
names(mdata)[7] <- "measure"
names(mdata)[8] <- "bet"
# change order levels
mdata$measure <- factor(mdata$measure ,
                       levels = c("samefeature", "bothfeatures", "nofeature", "difffeature"))


#BY SHARED FEATURES#-----------------------------------------------------
#compute mean correct across trials
ms <- aggregate(bet ~ speaker + context + measure , data=mdata, FUN=mean)
ms$cih <- aggregate(bet ~ speaker + context + measure , data=mdata,FUN=ci.high)$bet
ms$cil <- aggregate(bet ~ speaker + context + measure , data=mdata,FUN=ci.low)$bet

ggplot(ms, aes(x=context, y=bet, fill=speaker)) +
  ggtitle("generalization item") +
  geom_bar(stat="identity",position = "dodge",aes(fill=speaker))+
  geom_linerange(aes(ymin = bet-cil,
                     ymax = bet+cih),
                 size = 1.2,
                 position = position_dodge(.9)) +
  geom_hline(yintercept=50,lty=2) +
  scale_x_discrete(name="") +
  scale_y_continuous(limits = c(0,100),
                     name = "Mean Bets") +
  scale_fill_discrete(name="",
                      breaks=c("\"speaker\"", "\"no speaker\""),
                      labels=c("speaker", "no speaker")) +
  facet_grid(. ~ measure ) + 
  theme_bw(base_size=18) 


#PLOT BY CRITICAL FEATURE TYPE#---------------------------------

ms <- aggregate(bothfeatures ~ example, data=d,FUN=mean)
ms$cih <- aggregate(bothfeatures ~ example, data=d,FUN=ci.high)$bothfeatures
ms$cil <- aggregate(bothfeatures ~ example, data=d,FUN=ci.low)$bothfeatures

ggplot(ms, aes(x=example, y=bothfeatures)) +
  ggtitle("Critical Feature Manipulation: Both Feature Flower Generalization") +
  geom_bar(stat="identity",position = "dodge") +
  geom_linerange(aes(ymin = bothfeatures-cil,
                     ymax = bothfeatures+cih),
                 size = 1.2,
                 position = position_dodge(.9)) +
  geom_hline(yintercept=50,lty=2) +
  scale_y_continuous(limits = c(0,100),
                     name = "Mean Bets") +
  scale_x_discrete(name="") +
theme_bw(base_size=18) 
