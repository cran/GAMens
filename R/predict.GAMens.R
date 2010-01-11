predict.GAMens <- 
function(object, data,...) 
{
	formula <- object[[2]]
	fusion <- object[[8]]
	targets <- data[,as.character(formula[[2]])]  
	nclasses <- nlevels(targets)
	target_classes <- unique(targets)
	target_classes_s <- target_classes[order(target_classes)]
	newtargets <- as.numeric(data[,ncol(data)] == target_classes_s[2])
	newdata = cbind(data[,1:ncol(data)-1],newtargets)

	iter<-object[[3]]
	n <- length(newdata[,1])
	cutoff <- 0.5
	errors <- 1 - object[[12]]

	newdata_predictions <- data.frame(rep(0,n)) 
	predictions_p <- data.frame(cbind(1:nrow(newdata)))
	predictions_c <- data.frame(cbind(1:nrow(newdata)))
	for (m in 1:iter) {
		pred <- predict.gam(object[[1]][[m]],data,type="response")
		pred2 <- as.data.frame(as.numeric(pred))
		predictions_p <- cbind(predictions_p,pred2)
		pred2 <- as.data.frame(cbind(as.numeric(pred) > cutoff)*1)
		predictions_c <- cbind(predictions_c,pred2)

		temp_pred_p <- predictions_p
		temp_pred_p[is.na(temp_pred_p)] <- 0
		if (m > 1) {temp_sums <- rowSums(temp_pred_p[,2:ncol(temp_pred_p)])
			temp_n <- rowSums(!is.na(predictions_p[,2:ncol(temp_pred_p)]))

		}
		else { temp_sums <- temp_pred_p[,2:ncol(temp_pred_p)]
			temp_n <- !is.na(predictions_p[,2:ncol(temp_pred_p)])*1
		}
	}

	temp_pred_p <- predictions_p
	temp_pred_p[is.na(temp_pred_p)] <- 0
	temp_pred_c <- predictions_c
	temp_pred_c[is.na(temp_pred_c)] <- 0
	temp_sums <- rowSums(temp_pred_p[,2:ncol(temp_pred_p)])
	temp_sums_cl <- rowSums(temp_pred_c[,2:ncol(temp_pred_p)])
	temp_n <- rowSums(!is.na(predictions_p[,2:ncol(temp_pred_p)]))
	
	if (fusion == "avgagg") {
		pred <- cbind(temp_sums / temp_n)
		class <- as.numeric(pred > cutoff)
	}else if (fusion == "w.avgagg") {
		temp_sums_weighted <- as.matrix(temp_pred_p[,2:ncol(temp_pred_p)]) %*% (1 - errors)
		temp_n_weighted <- as.matrix(!is.na(predictions_p[,2:ncol(temp_pred_p)])*1) %*% (1 - errors)
		pred <- temp_sums_weighted / temp_n_weighted
		class <- as.numeric(pred > cutoff)
	}else if (fusion == "majvote") {
		pred <- cbind(temp_sums_cl / temp_n)
		class <- as.numeric(pred > cutoff)
	}else if (fusion == "w.majvote") {
		temp_sums_weighted <- as.matrix(temp_pred_c[,2:ncol(temp_pred_c)]) %*% (1 - errors)
		temp_n_weighted <- as.matrix(!is.na(predictions_c[,2:ncol(temp_pred_c)])*1) %*% (1 - errors)
		pred <- temp_sums_weighted / temp_n_weighted
		class <- as.numeric(pred > cutoff)}
	class <-  cbind(gsub(1,target_classes_s[[2]],class,fixed=FALSE))
	class <-  cbind(gsub(0,target_classes_s[[1]],class,fixed=FALSE))
	conf <- table(class, data[,as.character(formula[[2]])], dnn=c("Predicted Class", "Observed Class"))
	output <- list(pred=pred, class=class, conf=conf)
}
