require(FactoMineR)
require(factoextra)
library(data.table)
require(reshape2)

setwd("~/Google_Drive/Projects/Automated-Health-Responses/data")
df = fread("behavior_df.csv")

factor_columns <- colnames(df)


training <- df[ ,c(factor_columns),with=FALSE] 

for (c in colnames(df)){
  df[,which(colnames(df) %in% c)] =  as.factor(df[,which(colnames(df) %in% c)])
}
df = as.data.frame(df)


res.mca <- MCA(df)


fviz_mca_var(res.mca, col.var = "cos2",
             #gradient.cols = c("#00AFBB", "#E7B800", "#FC4E07"), 
             repel = FALSE, # Avoid text overlapping
             ggtheme = theme_minimal())
