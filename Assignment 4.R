library(tm)
library(SnowballC)
library(caret)
library(klaR)
library(kernlab) 

#Loading the dataset
farmads.df<-read.csv("~/Desktop/IIT/Predictive Analytics/Data files/FarmAds.csv")
str(farmads.df)
summary(farmads.df)
head(farmads.df)

#ConvertING sentences into a corpus
corp <- VCorpus(DataframeSource(farmads.df))

# Preprocess the text
farmads.df$label<-as.factor((farmads.df$label))

# 1. Convert to lowercase
corp <- tm_map(corp, content_transformer(tolower))

# 2. Remove punctuation
corp <- tm_map(corp, removePunctuation)

# 3. Remove stop words
corp <- tm_map(corp, removeWords, stopwords("english"))

# 4. Strip whitespace
corp <- tm_map(corp, stripWhitespace)

# 5. Apply stemming
corp <- tm_map(corp, stemDocument)

# Computing TD Matrix
tdm <- TermDocumentMatrix(corp)
inspect(tdm)
dtm <- DocumentTermMatrix(corp)
inspect(dtm)

dtm<-removeSparseTerms(dtm,0.94)

### Term Frequency--Inverse Document Frequency (TF-IDF)

dtm.matrix <- as.matrix(dtm)
newdata <- as.data.frame(dtm.matrix)
newdata$label <- farmads.df$label

# Split the data into training and testing sets
set.seed(1947)
idx<- createDataPartition(newdata$label, p=0.8, list=FALSE)  # From caret pkg
train.df <- newdata[idx, ]
test.df <- newdata[-idx,]


### run naive bayes using klaR package
farmads.nb<-NaiveBayes(label~.,data=train.df)
farmads.nb
confusionMatrix(predict(farmads.nb, newdata=test.df)$class, factor(test.df$label))

# training a simple SVM with linear kernel
farmads.svm <- ksvm(label ~ ., data = train.df, kernel = "vanilladot")
farmads.svm

# predictions on testing dataset
farmads.svm.pred <- predict(farmads.svm, test.df)

confusionMatrix(farmads.svm.pred, test.df$label)

## Improving model performance
# change to a RBF kernel
farmads.svm.rbf <- ksvm(label ~ ., data = train.df, kernel = "rbfdot")
farmads.svm.rbf <- predict(farmads.svm.rbf, test.df)
confusionMatrix(farmads.svm.rbf, test.df$label)
