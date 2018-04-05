#' Applies the GAMbag, GAMrsm or GAMens ensemble classifier to a data set
#'
#' Fits the GAMbag, GAMrsm or GAMens ensemble algorithms for binary
#' classification using generalized additive models as base classifiers.
#'
#' The \code{GAMens} function applies the GAMbag, GAMrsm or GAMens ensemble
#' classifiers (De Bock et al., 2010) to a data set. GAMens is the default with
#' (\code{bagging=TRUE} and \code{rsm=TRUE}. For GAMbag, \code{rsm} should be
#' specified as \code{FALSE}.  For GAMrsm, \code{bagging} should be
#' \code{FALSE}.
#'
#' The \code{GAMens} function provides the possibility for automatic formula
#' specification. In this case, dichotomous variables in \code{data} are
#' included as linear terms, and other variables are assumed continuous,
#' included as nonparametric terms, and estimated by means of smoothing
#' splines. To enable automatic formula specification, use the generic formula
#' \code{[response variable name]~.} in combination with \code{autoform =
#' TRUE}. Note that in this case, all variables available in \code{data} are
#' used in the model. If a formula other than \code{[response variable name]~.}
#' is specified then the \code{autoform} option is automatically overridden. If
#' \code{autoform=FALSE} and the generic formula \code{[response variable
#' name]~.} is specified then the GAMs in the ensemble will not contain
#' nonparametric terms (i.e., will only consist of linear terms).
#'
#' Four alternative fusion rules for member classifier outputs can be
#' specified. Possible values are \code{'avgagg'} for average aggregation
#' (default), \code{'majvote'} for majority voting, \code{'w.avgagg'} for
#' weighted average aggregation, or \code{'w.majvote'} for weighted majority
#' voting.  Weighted approaches are based on member classifier error rates.
#'
#' @param formula a formula, as in the \code{gam} function. Smoothing splines
#' are supported as nonparametric smoothing terms, and should be indicated by
#' \code{s}. See the documentation of \code{s} in the \code{gam} package for
#' its arguments. The \code{GAMens} function also provides the possibility for
#' automatic \code{formula} specification. See 'details' for more information.
#' @param data a data frame in which to interpret the variables named in
#' \code{formula}.
#' @param rsm_size an integer, the number of variables to use for random
#' feature subsets used in the Random Subspace Method. Default is 2.  If
#' \code{rsm=FALSE}, the value of \code{rsm_size} is ignored.
#' @param autoform if \code{FALSE} (default), the model specification in
#' \code{formula} is used. If \code{TRUE}, the function triggers automatic
#' \code{formula} specification. See 'details' for more information.
#' @param iter an integer, the number of base classifiers (GAMs) in the
#' ensemble. Defaults to \code{iter=10} base classifiers.
#' @param df an integer, the number of degrees of freedom (df) used for
#' smoothing spline estimation. Its value is only used when \code{autoform =
#' TRUE}. Defaults to \code{df=4}. Its value is ignored if a formula is
#' specified and \code{autoform} is \code{FALSE}.
#' @param bagging enables Bagging if value is \code{TRUE} (default). If
#' \code{FALSE}, Bagging is disabled. Either \code{bagging}, \code{rsm} or both
#' should be \code{TRUE}
#' @param rsm enables Random Subspace Method (RSM) if value is \code{TRUE}
#' (default). If \code{FALSE}, RSM is disabled. Either \code{bagging},
#' \code{rsm} or both should be \code{TRUE}
#' @param fusion specifies the fusion rule for the aggregation of member
#' classifier outputs in the ensemble. Possible values are \code{'avgagg'}
#' (default), \code{'majvote'}, \code{'w.avgagg'} or \code{'w.majvote'}.
#' @return An object of class \code{GAMens}, which is a list with the following
#' components: \item{GAMs}{the member GAMs in the ensemble.} \item{formula}{the
#' formula used tot create the \code{GAMens} object.  } \item{iter}{the
#' ensemble size. } \item{df}{number of degrees of freedom (df) used for
#' smoothing spline estimation. } \item{rsm}{indicates whether the Random
#' Subspace Method was used to create the \code{GAMens} object. }
#' \item{bagging}{indicates whether bagging was used to create the
#' \code{GAMens} object. } \item{rsm_size}{the number of variables used for
#' random feature subsets. } \item{fusion_method}{the fusion rule that was used
#' to combine member classifier outputs in the ensemble. } \item{probs}{the
#' class membership probabilities, predicted by the ensemble classifier.  }
#' \item{class}{the class predicted by the ensemble classifier. }
#' \item{samples}{an array indicating, for every base classifier in the
#' ensemble, which observations were used for training. } \item{weights}{a
#' vector with weights defined as (1 - error rate). Usage depends upon
#' specification of \code{fusion_method}. }
#' @export
#' @author Koen W. De Bock \email{kdebock@@audencia.com}, Kristof Coussement
#' \email{K.Coussement@@ieseg.fr} and Dirk Van den Poel
#' \email{Dirk.VandenPoel@@ugent.be}
#' @seealso \code{\link{predict.GAMens}}, \code{\link{GAMens.cv}}
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
#' @import mlbench caTools mlbench
#' @importFrom stats as.formula binomial predict terms
#' @examples
#'
#'
#' ## Load data (mlbench library should be loaded)
#' library(mlbench)
#' data(Ionosphere)
#' IonosphereSub<-Ionosphere[,c("V1","V2","V3","V4","V5","Class")]
#'
#' ## Train GAMens using all variables in Ionosphere dataset
#' Ionosphere.GAMens <- GAMens(Class~., IonosphereSub ,4 , autoform=TRUE,
#' iter=10 )
#'
#' ## Compare classification performance of GAMens, GAMrsm and GAMbag ensembles,
#' ## using 4 nonparametric terms and 2 linear terms
#' Ionosphere.GAMens <- GAMens(Class~s(V3,4)+s(V4,4)+s(V5,3)+s(V6,5)+V7+V8,
#' Ionosphere ,3 , autoform=FALSE, iter=10 )
#'
#' Ionosphere.GAMrsm <- GAMens(Class~s(V3,4)+s(V4,4)+s(V5,3)+s(V6,5)+V7+V8,
#' Ionosphere ,3 , autoform=FALSE, iter=10, bagging=FALSE, rsm=TRUE )
#'
#' Ionosphere.GAMbag <- GAMens(Class~s(V3,4)+s(V4,4)+s(V5,3)+s(V6,5)+V7+V8,
#' Ionosphere ,3 , autoform=FALSE, iter=10, bagging=TRUE, rsm=FALSE )
#'
#' ## Calculate AUCs (for function colAUC, load caTools library)
#' library(caTools)
#' GAMens.auc <- colAUC(Ionosphere.GAMens[[9]], Ionosphere["Class"]=="good",
#' plotROC=FALSE)
#' GAMrsm.auc <- colAUC(Ionosphere.GAMrsm[[9]], Ionosphere["Class"]=="good",
#' plotROC=FALSE)
#' GAMbag.auc <- colAUC(Ionosphere.GAMbag[[9]], Ionosphere["Class"]=="good",
#' plotROC=FALSE)
#'
GAMens <-
function(formula, data, rsm_size=2, autoform=FALSE, iter=10, df=4, bagging=TRUE, rsm=TRUE, fusion="avgagg")
{

	formula<- as.formula(formula)
	setdiff.data.frame <- function(A,B) A[ !duplicated( rbind(B,A) )[ -seq_len(nrow(B))] , ]
	if (!(as.character(fusion) %in% c("avgagg","majvote","w.avgagg","w.majvote"))){
		stop("fusion must be either 'avgagg', 'majvote', 'w.avgagg' or 'w.majvote' ")
	}
	if ((rsm == TRUE) & !(rsm_size > 0)){
		stop("Enter a valid random feature subspace size (rsm_size)")
	}
	if (!(rsm == TRUE) & !(bagging == TRUE)){
		stop("Either bagging, rsm or both must be set to TRUE")
	}
	if ((autoform == TRUE) & !(formula[3] == ".()")){
		warning("Autoform option will be overridden by the formula specification. In order to use autoform, set formula to '[outcome variable name]~.'")
		autoform <- FALSE
	}
	if ((autoform == FALSE) & (formula[3] == ".()")){
		warning("Autoform option is FALSE and a generic formula is used. The ensemble classifier will consist out of GAMs without nonparametric terms.")
		autoform <- FALSE
	}
	if ((rsm==TRUE) & (autoform == FALSE) & (!(formula[3] == ".()") & length(all.vars(formula[[3]]))<rsm_size)){
		stop("rsm_size should be smaller than the number of explanatory variables specified in formula")
	}

	targets <- data[,as.character(formula[[2]])]
	nclasses <- nlevels(targets)
	depvarname <- as.character(formula[[2]])

	target_classes <- unique(targets)
	target_classes_s <- target_classes[order(target_classes)]
	newtargets <- as.numeric(data[,ncol(data)] == target_classes_s[2])
	newdata = cbind(data[,1:ncol(data)-1],newtargets)
	names(newdata)[ncol(newdata)] <- depvarname


	n <- length(data[,1])
	if (formula[3] == ".()"){
		p <- (length(newdata[1,])-1)
 		varnames <- as.matrix(setdiff.data.frame(as.matrix(names(data)),as.matrix(depvarname)))
	}else {p <- nrow(as.matrix(all.vars(formula)))-1
		varnames <- as.matrix(setdiff.data.frame(as.matrix(cbind(all.vars(formula))),as.matrix(depvarname)))
		formula_terms <- attr(terms(formula),"term.labels")
	}

     	gam_models <- list()
	oob_predictions_p <- data.frame(cbind(1:nrow(newdata)))
	oob_predictions_c <- data.frame(cbind(1:nrow(newdata)))
	names(oob_predictions_p) <- "ID"
	names(oob_predictions_c) <- "ID"
	bootstraps <- array(0, c(n,iter))
	errors <- array(0,iter)

	id_added <- cbind(newdata,1:nrow(newdata))
	names(id_added)[ncol(id_added)] <- "ID"
	cutoff <- 0.5
	treshold <- 2
	for (m in 1:iter) {
		flag <- FALSE
		gam_model <- NA
		while (is.na(gam_model[1])) {
			if (rsm==TRUE) {rfs <- sample(1:p,rsm_size,replace=FALSE)} else {rfs <- 1:p}
			if (bagging==TRUE) {bootstrap<- sample(1:n,replace=TRUE)} else {bootstrap <- 1:n}
			if (formula[3] == ".()") {
				selectedvars <- names(newdata[bootstrap,rfs])
				if (autoform==TRUE) {
					rfs_linvars <- character(0)
					rfs_nparvars <- character(0)
					for (vn in 1:length(selectedvars)) {
						varname <- selectedvars[vn]
						uniq <- dim(unique(as.data.frame(data[bootstrap, rfs[vn]])))[1]
						if (uniq<=treshold) {rfs_linvars <- rbind(rfs_linvars,varname) }
						if (uniq>treshold) {rfs_nparvars <- rbind(rfs_nparvars,varname) }
					}
					if (length(rfs_nparvars > 0)) {
						npar_form1 <- paste("s(",rfs_nparvars,",",df,")")
						npar_form2 <- paste(npar_form1,collapse = "+")
					} else {npar_form2 <- character(0)}
					if (length(rfs_linvars > 0)) {
						lin_form <- paste(rfs_linvars,collapse= "+")
						if (length(rfs_nparvars > 0)) {finalstring = paste(depvarname,"~",npar_form2,"+",lin_form)}
						else {finalstring <- paste(depvarname,"~",lin_form)}
					} else {finalstring = paste(depvarname,"~",npar_form2)}
					fmla <- as.formula(finalstring)
				}
				if (autoform==FALSE) {
					lin_form <- paste(selectedvars,collapse= "+")
					finalstring = paste(depvarname,"~",lin_form)
					fmla <- as.formula(finalstring)
				}
			} else {
				selected_terms <- formula_terms[rfs]
				lin_form <- paste(selected_terms,collapse= "+")
				finalstring = paste(depvarname,"~",lin_form)
				fmla <- as.formula(finalstring)
			}

		try(gam_model <- gam(fmla,data=newdata[bootstrap,], family=binomial(link="logit")),silent=TRUE)
		}
		if (bagging == TRUE) {oob_data <- setdiff.data.frame(id_added,id_added[bootstrap,])} else {oob_data <- id_added[bootstrap,]}
		oob_cnt <- nrow(oob_data)
		oob_predict <- predict(gam_model,oob_data,type="response")
		oob_predict_r <- as.data.frame(cbind(as.numeric(oob_predict),oob_data[,"ID"]))
		names(oob_predict_r)[2] <- "ID"
		coln <- paste("pred",m,sep="")
		names(oob_predict_r)[1] <- coln
		oob_predict_c <- as.data.frame(cbind((oob_predict_r[,1] > cutoff),oob_predict_r[,2]))
		names(oob_predict_c)[2] <- "ID"
		names(oob_predict_c)[1] <- coln
		ind<-as.numeric(oob_data[depvarname] != oob_predict_c[,1])
		err<- sum(ind)/oob_cnt
		errors[m] <- err

		oob_predictions_p <- merge(oob_predictions_p,oob_predict_r, by = "ID", all= "TRUE")
		oob_predictions_c <- merge(oob_predictions_c,oob_predict_c, by = "ID", all= "TRUE")
		gam_models[[m]] <- gam_model
		bootstraps[,m]<-bootstrap

	}
	temp_oob_pred_p <- oob_predictions_p
	temp_oob_pred_p[is.na(temp_oob_pred_p)] <- 0
	temp_oob_pred_c <- oob_predictions_c
	temp_oob_pred_c[is.na(temp_oob_pred_c)] <- 0
	temp_sums <- rowSums(temp_oob_pred_p[,2:ncol(temp_oob_pred_p)])
	temp_sums_cl <- rowSums(temp_oob_pred_c[,2:ncol(temp_oob_pred_p)])
	temp_n <- rowSums(!is.na(oob_predictions_p[,2:ncol(temp_oob_pred_p)]))

	if (fusion == "avgagg") {
		pred <- cbind(temp_sums / temp_n)
		class <- as.numeric(pred > cutoff)
	}else if (fusion == "w.avgagg") {
		temp_sums_weighted <- as.matrix(temp_oob_pred_p[,2:ncol(temp_oob_pred_p)]) %*% (1 - errors)
		temp_n_weighted <- as.matrix(!is.na(oob_predictions_p[,2:ncol(temp_oob_pred_p)])*1) %*% (1 - errors)
		pred <- temp_sums_weighted / temp_n_weighted
		class <- as.numeric(pred > cutoff)
	}else if (fusion == "majvote") {
		pred <- cbind(temp_sums_cl / temp_n)
		class <- as.numeric(pred > cutoff)
	}else if (fusion == "w.majvote") {
		temp_sums_weighted <- as.matrix(temp_oob_pred_c[,2:ncol(temp_oob_pred_c)]) %*% (1 - errors)
		temp_n_weighted <- as.matrix(!is.na(oob_predictions_c[,2:ncol(temp_oob_pred_c)])*1) %*% (1 - errors)
		pred <- temp_sums_weighted / temp_n_weighted
		class <- as.numeric(pred > cutoff)}
	class <-  cbind(gsub(1,target_classes_s[[2]],class,fixed=FALSE))
	class <-  cbind(gsub(0,target_classes_s[[1]],class,fixed=FALSE))

	ans<- list(GAMs=gam_models, formula=fmla, iter=iter, df=df, rsm=rsm, bagging=bagging, rsm_size=rsm_size, fusion_method=fusion, probs=pred, class=class, samples=bootstraps, weights=1-errors)
	class(ans) <- "GAMens"
	ans
}
