% --- Source file: GAMens.cv.Rd ---
\name{GAMens.cv}
\alias{GAMens.cv}

\title{Runs v-fold cross validation with GAMbag, GAMrsm or GAMens ensemble classifier}

\description{
	In v-fold cross validation, the data are divided into \code{v} subsets of approximately equal size. Subsequently, 
	one of the \code{v} data parts is excluded while the remainder of the data is used to create a \code{GAMens} object.
	Predictions are generated for the excluded data part. The process is repeated \code{v} times.
}

\usage{
GAMens.cv(formula, data, cv, rsm_size=2, autoform=FALSE, iter=10, 
	df=4, bagging=TRUE, rsm=TRUE, fusion="avgagg") 

}


\arguments{
  \item{formula}{ a formula, as in the \code{gam} function. Smoothing splines are supported
   as nonparametric smoothing terms, and should be indicated by \code{s}. See the documentation of \code{s} in the
   \code{gam} package for its arguments. The \code{GAMens} function also provides the possibility for automatic
   \code{formula} specification. See 'details' for more information.  }
  \item{data}{a data frame in which to interpret the variables named in \code{formula}.  }
  \item{cv}{An integer specifying the number of folds in the cross-validation. }
  \item{rsm_size}{an integer, the number of variables to use for random feature subsets used in the Random Subspace Method. Default is 2.
   If \code{rsm=FALSE}, the value of \code{rsm_size} is ignored. }
  \item{autoform}{ if \code{FALSE} (by default), the model specification in \code{formula} is used. If \code{TRUE}, 
   the function triggers automatic \code{formula} specification. See 'details' for more information.  }
  \item{iter}{an integer, the number of base (member) classifiers (GAMs) in the ensemble. Defaults to \code{iter=10} 
   base classifiers.  }
  \item{df}{an integer, the number of degrees of freedom (df) used for smoothing spline estimation. Its value
   is only used when \code{autoform = TRUE}. Defaults to \code{df=4}. Its value is ignored if a formula is 
   specified and \code{autoform} is \code{FALSE}.}
  \item{bagging}{ enables Bagging if value is \code{TRUE} (default). If \code{FALSE}, 
   Bagging is disabled. Either \code{bagging}, \code{rsm} or both should be \code{TRUE}}
  \item{rsm}{ enables Random Subspace Method (RSM) if value is \code{TRUE} (default). If \code{FALSE}, 
   rsm is disabled. Either \code{bagging}, \code{rsm} or both should be \code{TRUE}}
  \item{fusion}{specifies the fusion rule for the aggregation of member classifier outputs in the ensemble. Possible values are
  \code{'avgagg'} for average aggregation (default), \code{'majvote'} for majority voting, \code{'w.avgagg'} for
  weighted average aggregation based on base classifier error rates, or \code{'w.majvote'} for weighted majority
  voting. }

}


\value{
   An object of class \code{GAMens.cv}, which is a list with the following components:
  \item{foldpred}{a data frame with, per fold, predicted class membership probabilities for the left-out observations. }
  \item{pred}{a data frame with predicted class membership probabilities. }
  \item{foldclass}{a data frame with, per fold, predicted classes for the left-out observations. }
  \item{class}{a data frame with predicted classes. }
  \item{conf}{the confusion matrix which compares the real versus predicted class memberships, based on the \code{class} object. }
}

\references{De Bock, K. W., Coussement, K. and Van den Poel, D. (2010): "Ensemble Classification based on generalized additive models". Computational Statistics & Data Analysis, doi:10.1016/j.csda.2009.12.013. 

  Breiman, L. (1996): "Bagging predictors". Machine Learning, Vol 24, 2, pp. 123--140.

  Hastie, T. and Tibshirani, R. (1990): "Generalized Additive Models", Chapman and Hall, London.

  Ho, T. K. (1998): "The random subspace method for constructing decision forests". IEEE Transactions on Pattern Analysis and Machine Intelligence, Vol 20, 8, pp. 832--844.  }

\author{Koen W. De Bock \email{Koen.DeBock@UGent.be}, Kristof Coussement \email{K.Coussement@Ieseg.fr} and Dirk Van den Poel \email{Dirk.VandenPoel@UGent.be} }


\seealso{ 
       \code{\link{predict.GAMens}},
       \code{\link{GAMens}} }



\examples{
## Load data: mlbench library should be loaded!)
library(mlbench)
data(Sonar)

## Perform 10-fold cross-validation of GAMrsm ensemble on Sonar data 
## using all variables
Sonar.cv.GAMrsm <- GAMens.cv(Class~., Sonar ,10, 3 , autoform=TRUE, iter=10, 
bagging=FALSE,rsm=TRUE )

## Compare classification performance of GAMens, GAMrsm and GAMbag 
## ensembles, using all variables in the Sonar dataset, based on 10-fold 
## cross validation runs
Sonar.cv.GAMens <- GAMens.cv(Class~s(V1,4)+s(V2,3)+s(V3,4)+V4+V5+V6, 
Sonar ,5, 4 , autoform=FALSE, iter=10 )

Sonar.cv.GAMrsm <- GAMens.cv(Class~s(V1,4)+s(V2,3)+s(V3,4)+V4+V5+V6, 
Sonar ,5, 4 , autoform=FALSE, iter=10, bagging=FALSE, rsm=TRUE )

Sonar.cv.GAMbag <- GAMens.cv(Class~s(V1,4)+s(V2,3)+s(V3,4)+V4+V5+V6, 
Sonar ,5, 4 , autoform=FALSE, iter=10, bagging=TRUE, rsm=FALSE )

## Calculate AUCs (for function colAUC, load caTools library)
library(caTools)
GAMens.cv.auc <- colAUC(Sonar.cv.GAMens[[2]], Sonar["Class"]=="R", 
plotROC=FALSE)
GAMrsm.cv.auc <- colAUC(Sonar.cv.GAMrsm[[2]], Sonar["Class"]=="R", 
plotROC=FALSE)
GAMbag.cv.auc <- colAUC(Sonar.cv.GAMbag[[2]], Sonar["Class"]=="R", 
plotROC=FALSE)

}


\keyword{models}
\keyword{classif}
