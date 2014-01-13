##contextandspeaker SURVEY Analysis
# Elise Sugarman December 2013
# Bar graphs

rm(list=ls())

#--LOAD PACKAGES--
library(ggplot2)
library(boot)
library(bootstrap)
library(reshape2)

#--READ IN DATA--
setwd("/Documents/GRADUATE_SCHOOL/Projects/pragLearn/Elise Project/data/") # set working directory and read in data 
d <- read.csv("flowerdata_SURVEY.csv",header=TRUE)

mdata <- melt(d, id=c("example","name", "condition" , "speaker", "context" , "generalizations", "check_question"))
names(mdata)[8] <- "measure"
names(mdata)[9] <- "fc"

mdata$measure <- factor(mdata$measure ,
                        levels = c("same_feature", "both_features", "no_feature", "diff_feature"))

#BY SHARED FEATURES#-----------------------------------------------------
#compute mean correct across trials
ms <- aggregate(fc ~ speaker + context + measure , data=mdata, FUN=mean)
ms$cih <- aggregate(fc ~ speaker + context + measure , data=mdata,FUN=ci.high)$fc
ms$cil <- aggregate(fc ~ speaker + context + measure , data=mdata,FUN=ci.low)$fc

ggplot(ms, aes(x=context, y=fc, fill=speaker)) +
  ggtitle("generalization item") +
  geom_bar(stat="identity",position = "dodge",aes(fill=speaker))+
  geom_linerange(aes(ymin = fc-cil,
                     ymax = fc+cih),
                 size = 1.2,
                 position = position_dodge(.9)) +
  geom_hline(yintercept=50,lty=2) +
  scale_x_discrete(name="") +
  scale_y_continuous(limits = c(0,1),
                     name = "Prop. Choice") +
  scale_fill_discrete(name="",
                      breaks=c("\"speaker\"", "\"no speaker\""),
                      labels=c("speaker", "no speaker")) +
  facet_grid(. ~ measure ) + 
  theme_bw(base_size=18) 

#SAME_FEATURE#-----------------------------------------------------

ms <- aggregate(same_feature ~ speaker + context, data= d,FUN=mean)
ms$cih <- aggregate(same_feature ~ speaker + context, data= d,FUN=ci.high)$same_feature
ms$cil <- aggregate(same_feature ~ speaker + context, data= d,FUN=ci.low)$same_feature

ggplot(ms, aes(x=context, y=same_feature, fill=speaker)) +
  ggtitle("Generalizations to Same Feature Flower") +
  geom_bar(stat="identity",position = "dodge",aes(fill=speaker))+
  geom_linerange(aes(ymin = same_feature-cil,
                     ymax = same_feature+cih),
                 size = 1.2,
                 position = position_dodge(.9)) +
  geom_hline(yintercept=50,lty=2) +
  scale_y_continuous(limits = c(0,1.0),
                     name = "Probability of Generalization") +
  scale_x_discrete(name="") +
  scale_fill_discrete(name="")
theme_bw(base_size=18) 

#DIFF_FEATURE#-----------------------------------------------------

ms <- aggregate(diff_feature ~ speaker + context, data=d,FUN=mean)
ms$cih <- aggregate(diff_feature ~ speaker + context, data=d,FUN=ci.high)$diff_feature
ms$cil <- aggregate(diff_feature ~ speaker + context, data=d,FUN=ci.low)$diff_feature

ggplot(ms, aes(x=context, y=diff_feature, fill=speaker)) +
  ggtitle("Generalizations to Different Feature Flower") +
  geom_bar(stat="identity",position = "dodge",aes(fill=speaker))+
  geom_linerange(aes(ymin = diff_feature-cil,
                     ymax = diff_feature+cih),
                 size = 1.2,
                 position = position_dodge(.9)) +
  geom_hline(yintercept=50,lty=2) +
  scale_y_continuous(limits = c(0,1.0),
                     name = "Probability of Generalization") +
  scale_x_discrete(name="") +
  scale_fill_discrete(name="")
theme_bw(base_size=18) 

#BOTH_FEATURE#-----------------------------------------------------

ms <- aggregate(both_features ~ speaker + context, data=d,FUN=mean)
ms$cih <- aggregate(both_features ~ speaker + context, data=d,FUN=ci.high)$both_features
ms$cil <- aggregate(both_features ~ speaker + context, data=d,FUN=ci.low)$both_features

ggplot(ms, aes(x=context, y=both_features, fill=speaker)) +
  ggtitle("Generalizations to Both Feature Flower") +
  geom_bar(stat="identity",position = "dodge",aes(fill=speaker))+
  geom_linerange(aes(ymin = both_features-cil,
                     ymax = both_features+cih),
                 size = 1.2,
                 position = position_dodge(.9)) +
  geom_hline(yintercept=50,lty=2) +
  scale_y_continuous(limits = c(0,0.25),
                     name = "Probability of Generalization") +
  scale_x_discrete(name="") +
  scale_fill_discrete(name="")
theme_bw(base_size=18) 

#PLOT BY CRITICAL FEATURE TYPE#---------------------------------

ms <- aggregate(both_features ~ example, data=d,FUN=mean)
ms$cih <- aggregate(both_features ~ example, data=d,FUN=ci.high)$both_features
ms$cil <- aggregate(both_features ~ example, data=d,FUN=ci.low)$both_features

ggplot(ms, aes(x=example, y=both_features)) +
  ggtitle("Generalizations to Both Feature Flower") +
  geom_bar(stat="identity",position = "dodge") +
  geom_linerange(aes(ymin = both_features-cil,
                     ymax = both_features+cih),
                 size = 1.2,
                 position = position_dodge(.9)) +
  geom_hline(yintercept=50,lty=2) +
  scale_y_continuous(limits = c(0,0.25),
                     name = "Probability of Generalization") +
  scale_x_discrete(name="")
theme_bw(base_size=18) 