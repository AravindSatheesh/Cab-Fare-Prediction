#This statement is used to clear the environment
rm(list=ls(all=T))

#The libraries are loaded
print("The required libraries are installed and loaded")
x=c("ggplot2","DMwR","corrgram","rpart","randomForest")

#This statement is used to install packages from CRAN repository.
#repos attribute is used to denote the path from which libraries must be loaded.For running from command prompt, it is needed
install.packages(x, repos = "http://cran.us.r-project.org")

#This statement will return true if the library is found and is loaded successfully 
lapply(x,require,character.only=T)
rm(x) #The x variable is removed to free up space

#The working directory is set
setwd("C:\\Users\\admin\\Desktop\\Aravi\\Data Science\\Important Notes\\EdWisor\\Assignments\\Project 2 - Cab Fare Prediction")

#Now the data is loaded
#Here on analysing the data, in fare_amount variable, a value '430-' is mistakenly loaded. So it is also transformed into NA. 
#Even zeros are transformed into NA as they can be imputed.
#Even the date is in irregular format in 1327th row. So let us remove the row to avoid confusion.

print("The dataset is loaded")
cab_data = read.csv("train_cab.csv", header=T, na.strings=c(" ","","NA","430-"))
cab_data[cab_data == 0]= NA
cab_data=cab_data[-1328,]

############################# Missing Value Analysis ################################################

df= cab_data
missing_val = data.frame(apply(cab_data,2,function(x){sum(is.na(x))}))
missing_val$Columns = row.names(missing_val)
names(missing_val)[1] =  "Missing_percentage"
missing_val$Missing_percentage = (missing_val$Missing_percentage/nrow(cab_data)) * 100
missing_val = missing_val[order(-missing_val$Missing_percentage),]
row.names(missing_val) = NULL
missing_val = missing_val[,c(2,1)]
print("Before missing value analysis:")
print(missing_val)

#Only less than 2% of data are missing and so they can be imputed using one among the following methods.
#1. Central Tendency(Mean, Median, Mode)
#2. KNN Imputation

#Let us do trial and error for passenger_count variable and select the best method for this dataset.Let us take 280th row value for testing.

print("Missing Value Analysis")
print("Trial and Error")
print(paste("Actual Value:",cab_data[280,7]))

#Actual value = 2

#Median method
cab_data = df
cab_data[cab_data == 0]= NA
cab_data[280,7]=NA
cab_data$passenger_count[is.na(cab_data$passenger_count)] = median(cab_data$passenger_count, na.rm = T)
print(paste("Median Imputed Value:",cab_data[280,7]))

#Median Imputed Value: 1

#Mode method
getmode <- function(v) {
  uniqv <- unique(v)
  uniqv[which.max(tabulate(match(v, uniqv)))]
}
cab_data = df
cab_data[cab_data == 0]= NA
cab_data[280,7]=NA
cab_data$passenger_count[is.na(cab_data$passenger_count)] = getmode(cab_data$passenger_count)
print(paste("Mode Imputed Value:",cab_data[280,7]))

#Mode Imputed Value: 1

#KNN Imputation method
cab_data = df
cab_data[cab_data == 0]= NA
cab_data[280,7]=NA
cab_data_new=knnImputation(cab_data, k=3)
cab_data_new$passenger_count=round(cab_data_new$passenger_count)
print(paste("Knn Imputed Value:",cab_data_new[280,7]))

#Knn Imputed Value: 2

#From these three methods, Knn imputation is better.
#So the new values imputed by knn imputation method and used for further processing.
print("Knn imputed value is better and so it is used for imputing missing values here")

#Values after missing value analysis

missing_val = data.frame(apply(cab_data_new,2,function(x){sum(is.na(x))}))
missing_val$Columns = row.names(missing_val)
names(missing_val)[1] =  "Missing_percentage"
missing_val$Missing_percentage = (missing_val$Missing_percentage/nrow(cab_data_new)) * 100
missing_val = missing_val[order(-missing_val$Missing_percentage),]
row.names(missing_val) = NULL
missing_val = missing_val[,c(2,1)]
print("After missing value analysis:")
print(missing_val)

#Now the missing values are handled and we can move on to the next section.

################################# Outlier Analysis #################################################

print("Outlier Analysis is in process:")
#Creating Box and Whisker plots to check for outliers
for (i in c(1,3,4,5,6,7))
{
  assign(colnames(cab_data)[i],
         ggplot(cab_data,aes_string(x=cab_data[,i],y=colnames(cab_data)[i]))+
           labs(x=colnames(cab_data)[i])+
           stat_boxplot(geom = "errorbar", width = 0.5) +
           geom_boxplot(outlier.colour="red",fill = "grey" ,outlier.shape=18, outlier.size=1, notch=F))
}

gridExtra::grid.arrange(fare_amount, passenger_count, ncol=2)
gridExtra::grid.arrange(pickup_longitude, pickup_latitude, ncol=2)
gridExtra::grid.arrange(dropoff_longitude, dropoff_latitude, ncol=2)

#First let us make only the most extreme values as NA
#Then impute the values and check for outliers again

cab_data$fare_amount[cab_data$fare_amount>1000]=NA
cab_data$pickup_longitude[cab_data$pickup_longitude>-60]=NA
cab_data$pickup_latitude[cab_data$pickup_latitude>100]=NA
cab_data$pickup_latitude[cab_data$pickup_latitude<10]=NA
cab_data$dropoff_latitude[cab_data$dropoff_latitude<20]=NA
cab_data$dropoff_longitude[cab_data$dropoff_longitude>-60]=NA
cab_data$passenger_count[cab_data$passenger_count>7]=NA

#Handling Outliers using KNN Imputation method

#Taking a copy of the data and initiating columns for outlier handling
df = cab_data
df = knnImputation(df, k = 3)

#Creating Box and Whisker plots to check for outliers after basic outlier analysis
for (i in c(1,3,4,5,6,7))
{
  assign(colnames(df)[i],
         ggplot(df,aes_string(x=df[,i],y=colnames(df)[i]))+
           labs(x=colnames(df)[i])+
           stat_boxplot(geom = "errorbar", width = 0.5) +
           geom_boxplot(outlier.colour="red",fill = "grey" ,outlier.shape=18, outlier.size=1, notch=F))
}

gridExtra::grid.arrange(fare_amount, passenger_count, ncol=2)
gridExtra::grid.arrange(pickup_longitude, pickup_latitude, ncol=2)
gridExtra::grid.arrange(dropoff_longitude, dropoff_latitude, ncol=2)

cab_data = df

#After handling the basic outliers, let us find the distance travelled using latitude and longitude points using haversine formula
#Then handle the remaining outliers for efficient processing. 

#Calculate the great circle distance between two points on the earth (specified in decimal degrees)

haversine <- function(lon1, lat1, lon2, lat2){
   
  # convert decimal degrees to radians 
  radian <- function(deg){
    rad = (deg*22)/(7*180)
    return (rad)
  }

  lon1 = radian(lon1)
  lat1 = radian(lat1)
  lon2 = radian(lon2)
  lat2 = radian(lat2)
  

  # haversine formula 
  dlon = lon2 - lon1 
  dlat = lat2 - lat1 
  a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
  c = 2 * asin(sqrt(a)) 
  r = 6371 # Radius of earth in kilometers. Use 3956 for miles
  return (c * r)
}

#distance travelled is measured and added in the dataset
cab_data$distance_travelled = haversine(cab_data$pickup_longitude,cab_data$pickup_latitude,cab_data$dropoff_longitude,cab_data$dropoff_latitude)

#Now timestamp data is converted from string to required format and date and time data are fetched.
cab_data$pickup_datetime = strptime(cab_data$pickup_datetime, "%Y-%m-%d %H:%M:%S UTC")
cab_data$Year = format(cab_data$pickup_datetime,"%Y")
cab_data$Month = format(cab_data$pickup_datetime,"%m")
cab_data$Day = format(cab_data$pickup_datetime,"%d")
cab_data$Hour = format(cab_data$pickup_datetime,"%H")
cab_data$Time = format(cab_data$pickup_datetime,"%H:%M:%S")

#Now let us remove the latitude and longitude variables and pickup_datetime as new variables have been derived from them.

cab_data = cab_data[,-c(2,3,4,5,6)]
cab_data = cab_data[,c(1,3,2,4,5,6,7,8)]

#Now, again outlier analysis is carried upon the resulting datatset on fare_amount and distance_travelled variables

#Taking a copy of the data and initiating columns for outlier handling
df = cab_data
cnames = c("fare_amount","distance_travelled")

#Replace outliers with NA
print("Final outliers:")
for(i in cnames){
  val = df[,i][df[,i] %in% boxplot.stats(df[,i])$out]
  print(i)
  print(length(val)) #To visualise th total number of outliers
  df[,i][df[,i] %in% val] = NA
}

#Impute missing values using KNN
for (i in c(1,2,3,4,5,6)){
  df[,i]=as.numeric(df[,i])
}
df[,c(1,2,3,4,5,6)] = knnImputation(df[,c(1,2,3,4,5,6)], k = 3)


#Creating Box and Whisker plots to check for outliers after formatting
for (i in c(1,2)){
  assign(colnames(df)[i],
         ggplot(df,aes_string(x=df[,i],y=colnames(df)[i]))+
           labs(x=colnames(df)[i])+
           stat_boxplot(geom = "errorbar", width = 0.5) +
           geom_boxplot(outlier.colour="red",fill = "grey" ,outlier.shape=18, outlier.size=1, notch=F))
}

gridExtra::grid.arrange(fare_amount, distance_travelled, ncol=2)

#These outliers are very close to original range and can be retained.
#outliers have been efficiently handled and we can move on to the next preprocessing technique

#################################### Feature Selection ############################################

#Formation of Correlation matrix and heatmap based on the correlation between variables

print ("Feature Selection is in progress:")

corrgram(df[,], order = F,
         upper.panel=panel.pie, text.panel=panel.txt, main = "Correlation Plot")

#ANOVA test is carried out for category data
anova_test = aov(fare_amount~passenger_count+Year+Month+Day+Hour, data = df)

print(summary(anova_test))

#From the heatmap and ANOVA test, the variables 'passenger_count', 'Date', 'Time' are dropped.

df = subset(df, select = c(distance_travelled,Year,Month,fare_amount))

#Now let us move on to Feature Scaling technique.

####################################### Feature Scaling #########################################

#The variables 'distance_travelled', 'Year', 'Month' are to be normalised

print ("Feature Scaling is in progress:")
norm_var = c("distance_travelled","Year","Month")
for (i in norm_var){
  hist(df[,i])
  df[,i] = (df[,i] - min(df[,i]))/
    (max(df[,i] - min(df[,i])))
}
print("Data after pre-processing:")
print(head(df))

####################################### Modeling ####################################################

####################################### Error Metrics ###############################################

#RMSE
rmse = function(act, pred){
  (mean((act-pred)**2))**0.5
}

#R-squared
rsq = function(act, pred){
  (1-(sum((act-pred)**2)/sum((mean(act)-pred)**2)))
}

#Adjusted R-squared
adjrsq = function(rsq,n,k){
  (1-((1-rsq)*(n-1)/(n-k-1)))
}

###################################### Linear Regression #############################################

#The first model that we are using is the Linear Regression

train_index = sample(1:nrow(df),0.8*nrow(df)) #80% of the whole index is sampled for selecting the train data
#Train and test data are sampled
train_data = df[train_index,]
test_data = df[-train_index,]

#Model generation
reg_model = lm(fare_amount ~., data = train_data)

#Using the model, the dependent variable is predicted
reg_prediction = predict(reg_model, test_data[,1:3])

#Model Evaluation

print("Linear Regression Status:")
print(paste("RMSE - ",(rmse(test_data[,4],reg_prediction))))
rsquared = rsq(test_data[,4],reg_prediction)
print(paste("R-squared - ",rsquared))
print(paste("Adjusted R-squared - ",(adjrsq(rsquared,nrow(df),ncol(df)))))

#The outputs obtained are given as comments

#Linear Regression Status:
#"RMSE -  2.42711434030553"
#"R-squared -  0.824149623495949"
#"Adjusted R-squared -  0.82405601777364"


#To visualise the summary of the model
summary(reg_model)

##################################### Decision Tree ###########################################

#We can use the same train and test data and build and evaluate the model

#Model generation
tree_model = rpart(fare_amount ~., data = train_data, method="anova")

#Using the model, the dependent variable is predicted
tree_prediction = predict(tree_model, test_data[,1:3])

#Model Evaluation

print("Decision Tree Status:")
print(paste("RMSE - ",(rmse(test_data[,4],tree_prediction))))
rsquared = rsq(test_data[,4],tree_prediction)
print(paste("R-squared - ",rsquared))
print(paste("Adjusted R-squared - ",(adjrsq(rsquared,nrow(df),ncol(df)))))

#The outputs obtained are given as comments

#Decision Tree Status:
#"RMSE -  2.44292817637278"
#"R-squared -  0.617644287465166"
#"Adjusted R-squared - 0.617549061585697"

########################################## Random Forest ###########################################

#We can use the same train and test data and build and evaluate the model here too

#Model generation
forest_model = randomForest(fare_amount ~., data = train_data, ntree=10)

#Using the model, the dependent variable is predicted
forest_prediction = predict(forest_model, test_data[,1:3])

#Model Evaluation

print("Decision Tree Status:")
print(paste("RMSE - ",(rmse(test_data[,4],forest_prediction))))
rsquared = rsq(test_data[,4],forest_prediction)
print(paste("R-squared - ",rsquared))
print(paste("Adjusted R-squared - ",(adjrsq(rsquared,nrow(df),ncol(df)))))

#The outputs obtained are given as comments

#Random Forest Status:
#"RMSE -  2.48521972088313"
#"R-squared -  0.473395135048051"
#"Adjusted R-squared - 0.473263983845772"

#Based on the error metrics, for this dataset, Linear Regression model performs better when compared to all the other models.
#So Linear Regression model is used for future predictions.

######################################### Test Output ########################################

#Test data is invoked
test = read.csv("test.csv", header = T)
df = test

#distance travelled is measured
df$distance_travelled = haversine(df$pickup_longitude,df$pickup_latitude,df$dropoff_longitude,df$dropoff_latitude)

#Year and Month data are fetched from timestamp
df$pickup_datetime = strptime(df$pickup_datetime, "%Y-%m-%d %H:%M:%S UTC")
df$Year = format(df$pickup_datetime,"%Y")
df$Month = format(df$pickup_datetime,"%m")

df = df[,c('distance_travelled','Year','Month')]

#Normalising the data
norm_var = c("distance_travelled","Year","Month")
for (i in norm_var){
  df[,i] = as.numeric(df[,i])
  df[,i] = (df[,i] - min(df[,i]))/
    (max(df[,i] - min(df[,i])))
}

#Predicting the fare based on Regression Model
fare = predict(reg_model,df)
test$fare_amount = fare

#Storing the output in R Test Output.csv file
write.csv(test,"R Test Output.csv",row.names = FALSE)

print("The test data is invoked and fare is predicted")
print(head(test))

#Thus the regression model has predicted the fare_amount based on latitude longitude and timestamp details.