#' Runs v-fold cross validation with GAMbag, GAMrsm or GAMens ensemble
#' classifier
#'
#' In v-fold cross validation, the data are divided into \code{v} subsets of
#' approximately equal size. Subsequently, one of the \code{v} data parts is
#' excluded while the remainder of the data is used to create a \code{GAMens}
#' object.  Predictions are generated for the excluded data part. The process
#' is repeated \code{v} times.
#'
#'
#' @param formula a formula, as in the \code{gam} function. Smoothing splines
#' are supported as nonparametric smoothing terms, and should be indicated by
#' \code{s}. See the documentation of \code{s} in the \code{gam} package for
#' its arguments. The \code{GAMens} function also provides the possibility for
#' automatic \code{formula} specification. See 'details' for more information.
#' @param data a data frame in which to interpret the variables named in
#' \code{formula}.
#' @param cv An integer specifying the number of folds in the cross-validation.
#' @param rsm_size an integer, the number of variables to use for random
#' feature subsets used in the Random Subspace Method. Default is 2.  If
#' \code{rsm=FALSE}, the value of \code{rsm_size} is ignored.
#' @param autoform if \code{FALSE} (by default), the model specification in
#' \code{formula} is used. If \code{TRUE}, the function triggers automatic
#' \code{formula} specification. See 'details' for more information.
#' @param iter an integer, the number of base (member) classifiers (GAMs) in
#' the ensemble. Defaults to \code{iter=10} base classifiers.
#' @param df an integer, the number of degrees of freedom (df) used for
#' smoothing spline estimation. Its value is only used when \code{autoform =
#' TRUE}. Defaults to \code{df=4}. Its value is ignored if a formula is
#' specified and \code{autoform} is \code{FALSE}.
#' @param bagging enables Bagging if value is \code{TRUE} (default). If
#' \code{FALSE}, Bagging is disabled. Either \code{bagging}, \code{rsm} or both
#' should be \code{TRUE}
#' @param rsm enables Random Subspace Method (RSM) if value is \code{TRUE}
#' (default). If \code{FALSE}, rsm is disabled. Either \code{bagging},
#' \code{rsm} or both should be \code{TRUE}
#' @param fusion specifies the fusion rule for the aggregation of member
#' classifier outputs in the ensemble. Possible values are \code{'avgagg'} for
#' average aggregation (default), \code{'majvote'} for majority voting,
#' \code{'w.avgagg'} for weighted average aggregation based on base classifier
#' error rates, or \code{'w.majvote'} for weighted majority voting.
#' @return An object of class \code{GAMens.cv}, which is a list with the
#' following components: \item{foldpred}{a data frame with, per fold, predicted
#' class membership probabilities for the left-out observations. }
#' \item{pred}{a data frame with predicted class membership probabilities. }
#' \item{foldclass}{a data frame with, per fold, predicted classes for the
#' left-out observations. } \item{class}{a data frame with predicted classes. }
#' \item{conf}{the confusion matrix which compares the real versus predicted
#' class memberships, based on the \code{class} object. }
#' @author Koen W. De Bock \email{kdebock@@audencia.com}, Kristof Coussement
#' \email{K.Coussement@@ieseg.fr} and Dirk Van den Poel
#' \email{Dirk.VandenPoel@@ugent.be}
#' @seealso \code{\link{predict.GAMens}}, \code{\link{GAMens}}
#' @references De Bock, K.W. and Van den Poel, D. (2012):
#' "Reconciling Performance and Interpretability in Customer Churn Prediction Modeling
#' Using Ensemble Learning Based on Generalized Additive Models".
#' Expert Systems With Applications, Vol 39, 8, pp. 6816--6826.
#'
#' De Bock, K. W., Coussement, K. and Van den Poel, D. (2010):
#' "Ensemble Classification based on generalized additive models".
#' Computational Statistics & Data Analysis, Vol 54, 6, pp. 1535--1546.
#'
#' Breiman, L. (1996): "Bagging predictors". Machine Learning, Vol 24, 2, pp.
#' 123--140.
#'
#' Hastie, T. and Tibshirani, R. (1990): "Generalized Additive Models", Chapman
#' and Hall, London.
#'
#' Ho, T. K. (1998): "The random subspace method for constructing decision
#' forests". IEEE Transactions on Pattern Analysis and Machine Intelligence,
#' Vol 20, 8, pp. 832--844.
#' @keywords models classif
#' @export
#' @examples
#'
#' ## Load data: mlbench library should be loaded!)
#' library(mlbench)
#' data(Sonar)
#' SonarSub<-Sonar[,c("V1","V2","V3","V4","V5","V6","Class")]
#'
#' ## Obtain cross-validated classification performance of GAMrsm
#' ## ensembles, using all variables in the Sonar dataset, based on 5-fold
#' ## cross validation runs
#'
#' Sonar.cv.GAMrsm <- GAMens.cv(Class~s(V1,4)+s(V2,3)+s(V3,4)+V4+V5+V6,
#' SonarSub ,5, 4 , autoform=FALSE, iter=10, bagging=FALSE, rsm=TRUE )
#'
#' ## Calculate AUCs (for function colAUC, load caTools library)
#' library(caTools)
#'
#' GAMrsm.cv.auc <- colAUC(Sonar.cv.GAMrsm[[2]], SonarSub["Class"]=="R",
#' plotROC=FALSE)
#'
#'
GAMens.cv <-
function (formula, data, cv, rsm_size=2, autoform=FALSE, iter=10, df=4, bagging=TRUE, rsm=TRUE, fusion="avgagg")
{
	setdiff.data.frame <- function(A,B) A[ !duplicated( rbind(B,A) )[ -seq_len(nrow(B))] , ]
	n <- dim(data)[1]
	if(cv <2 | cv>n) {stop("The number of cross-validations should be larger than 2 and smaller than the number of observations")}

	bootstrap <- sample(1:n,replace=FALSE)
	obs_per_fold <- floor(n/cv)
	pred_per_fold <- as.data.frame(array(NA, c(n,cv)))
	pred <- as.data.frame(array(NA, c(n,1)))
	class_per_fold <- as.data.frame(array(NA, c(n,cv)))
	class <- as.data.frame(array(NA, c(n,1)))

    	for (c in 1:cv) {
        	if (c < cv) {fold <- data[bootstrap[(((c-1)*obs_per_fold)+(1:obs_per_fold))],]
		} else {fold <- data[bootstrap[(((c-1)*obs_per_fold):nrow(data))],]}
        	if (c < cv) {fold_ids <- bootstrap[(((c-1)*obs_per_fold)+(1:obs_per_fold))]
		} else {fold_ids <- bootstrap[(((c-1)*obs_per_fold):nrow(data))]}
		traindata_folds <- setdiff.data.frame(data,fold)
		GAMens_object <- GAMens(formula,traindata_folds, rsm_size, autoform=autoform,iter=iter,df=df,bagging=bagging,rsm=rsm,fusion=fusion)
		results <- predict(GAMens_object,fold)
		pred_per_fold[fold_ids,c] <- results[[1]]
		names(pred_per_fold)[[c]] <- paste("fold",c,sep="")
		pred[fold_ids,1] <- results[[1]]
		class_per_fold[fold_ids,c] <- results[[2]]
		names(class_per_fold)[[c]] <- paste("fold",c,sep="")
		class[fold_ids,1] <- results[[2]]
    	}

	conf <- table(as.matrix(class), data[,as.character(formula[[2]])], dnn=c("Predicted Class", "Observed Class"))
	output<- list(foldpred=pred_per_fold,pred=pred,foldclass=class_per_fold,class=class,conf=conf)
}
