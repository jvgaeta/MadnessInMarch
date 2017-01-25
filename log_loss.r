library(data.table); library(dplyr); library(reshape)

TourneyCompactResults <- fread("input/TourneyCompactResults.csv")

tc1 <- TourneyCompactResults %>% filter(Wteam < Lteam) %>% mutate(Id = paste(Season, "_", Wteam, "_", Lteam, sep=""), outcome=1, 
       			     	 	      	       	   	  team1=Wteam, team2=Lteam)
                                                                  
tc2 <- TourneyCompactResults %>% filter(Wteam > Lteam) %>% mutate(Id = paste(Season, "_", Lteam, "_", Wteam, sep=""), outcome=0,
                                                                  team1=Lteam, team2=Wteam)
results <- rbind(tc1, tc2) %>% select(Id, outcome, team1, team2, Season, Daynum)                                                              
kaggle.results <- results %>% filter(Season>=2012 & Daynum>=136) 

LogLoss <- function(pred, res){
  (-1/length(pred)) * sum (res * log(pred) + (1-res)*log(1-pred))
}

Accuracy <- function(pred, res) {
	count = 0
	isit = 0
	for (i in 1:length(pred)) {
		if (pred[i] >= 0.5) {
			isit = 1
		}
		else {
			isit = 0
		}
		if (isit == res[i]) {
			count = count + 1
		}
	}
	count / length(pred)
}

submission <- read.csv('submission.csv', stringsAsFactors=FALSE)
inner_join(submission, kaggle.results) %>% summarize(LogLoss(Pred, outcome))
inner_join(submission, kaggle.results) %>% summarize(Accuracy(Pred, outcome))
