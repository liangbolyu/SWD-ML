#constant
library(haven)
library(dplyr)
library(np)
library(ggplot2)
library(tidyverse)
library(SuperLearner)
library(randomForest)
library(xgboost)
library(e1071)
library(rpart)
library(nnet)
library(Matrix)
library(glmnet)

data=read.csv("")
data=data[order(data$id),]
data=data[order(data$time),]
data=data[order(data$cluster),]
d <- data[!duplicated(data[c("trained", "cluster")]),]
Z=d$trained
dataf=data[,-5]
dataf=dataf[,-5]
J=max(dataf$time)
Ni=as.numeric(table(dataf$cluster))/J
I=length(Ni)
clusteru=unique(dataf$cluster)
id=unique(dataf$id)
Dmu<-map(1:I, function(i){
  l=rep(1,Ni[i])
  D=rep(0,J)
  
  mu=rep(0,J)
  for(j in 1:J){
    D[j]=ifelse(Z[i]<=j,1,0)
    mu[j]=sum(Z<=j)/length(Z)
  }
  re=D-mu
  return(kronecker(re,l))
})


m=5
samplesplit=function(m,Z){
  I=length(Z)
  J=max(Z)
  ssp=rep(0,I)
  for(j in 1:J){
    l=which(Z==j)
    s=sample(rep(1:m, length.out = length(l)))
    ssp[l]=s
  }
  return(ssp)
}
samples=samplesplit(m,Z)
dataf$sp=1
dataf$sp <- samples[ match(dataf$cluster, clusteru) ]
dataf$g=1
for(i in 1:m){
  for(j in 1:J){
    train=which(dataf$sp!=i&dataf$time==j)
    Ytrain=dataf$y[train]
    Xtrain=dataf[train,5:8]
    test=which(dataf$sp==i&dataf$time==j)
    Ytest= dataf$y[test]
    Xtest=dataf[test,5:8]
    
    
    traindata=data.frame(y=Ytrain,Xtrain)
    
    #unadjusted
    model=lm(y~.,data=traindata)
    ypred=predict(model,newdata = Xtest)
    
    #linear regression
    model=lm(y~.,data=traindata)
    ypred=predict(model,newdata = Xtest)
    
    
    #superlearner 
    learners <- c("SL.glm", "SL.randomForest", "SL.rpart", "SL.svm")
    SL_model <- SuperLearner(Y = Ytrain,X = Xtrain,family = gaussian(),
                             SL.library = learners,
                             method = "method.NNLS")
    ypred <- predict(SL_model, newdata = Xtest)$pred
    
    dataf$g[test]=ypred
  }
}

dataf$residual=dataf$y-dataf$g

re=dataf$residual




resi_listf=split(re,dataf$cluster)
Dsquare=Reduce("+",Map(function(a, b) t(a) %*% b, Dmu, Dmu)) 
Dsinv=solve(Dsquare)
Dvsresidual=Reduce("+",Map(function(a, b) t(a) %*% b, Dmu, resi_listf)) 
betaest=Dsinv%*%Dvsresidual
phi=Map(function(a, b) {t(a) %*% (b-a%*%betaest)}, Dmu, resi_listf)
meat=Reduce("+",Map(function(a, b) a %*% t(b), phi, phi)) 
vmatrix=Dsinv%*%meat%*%Dsinv

betaest1=betaest
betaest1
vmatrix


num <- length(unique(dataf$time))

l1 <- diag(1, num)          
l2 <- matrix(1, num, num)   

Q1 <- rep(list(l1), I)
Q2 <- rep(list(l2), I)
Qfunction=function(i,t=1){
  q=list(Q1[[i]],Q2[[i]])
  return(q)
}
for(sss in 1:2){
  phiiold=map(1:I, function(i) {
    Q=Qfunction(i)
    phii=lapply(Q, function(Qq){
      t(Dmu[[i]])%*%Qq%*%(resi_listf[[i]]-Dmu[[i]]%*%betaest)
    })
    phii=do.call(rbind,phii)
    return (phii)
  })
  
  CI=Reduce("+",Map(function(a, b) a %*% t(b), phiiold, phiiold))
  CIinverse=solve(CI)
  phibeta=map(1:I, function(i) {
    Q=Qfunction(i)
    phii=lapply(Q, function(Qq){
      t(Dmu[[i]])%*%Qq%*%Dmu[[i]]
    })
    phii=do.call(rbind,phii)
    return (phii)
  })
  phibeta=Reduce("+",phibeta)
  phivarinverse=t(phibeta)%*%CIinverse%*%phibeta
  phivar=solve(phivarinverse)
  sqrt(phivar)
  phis=map(1:I, function(i) {
    Q=Qfunction(i)
    phii=lapply(Q, function(Qq){
      t(Dmu[[i]])%*%Qq%*%resi_listf[[i]]
    })
    phii=do.call(rbind,phii)
    return (phii)
  })
  phis=Reduce("+",phis)
  betaest=phivar%*%t(phibeta)%*%CIinverse%*%phis
  betaest}
betaest
phivar



#duration
library(haven)
library(dplyr)
library(np)
library(ggplot2)
library(tidyverse)
library(SuperLearner)
library(randomForest)
library(xgboost)
library(e1071)
library(rpart)
library(nnet)
library(Matrix)
library(glmnet)




data=read.csv("")
data=data[order(data$id),]
data=data[order(data$time),]
data=data[order(data$cluster),]
d <- data[!duplicated(data[c("trained", "cluster")]),]
Z=d$trained
dataf=data[,-5]
dataf=dataf[,-5]
J=max(dataf$time)
Ni=as.numeric(table(dataf$cluster))/J
I=length(Ni)
clusteru=unique(dataf$cluster)
id=unique(dataf$id)
Dmu<-map(1:I, function(i){
  l=rep(1,Ni[i])
  D=matrix(0,J,J)
  
  mu=matrix(0,J,J)
  cu=cumsum(pi)
  for(s1 in 1:J){
    for(s2 in 1:s1){
      D[s1,s2]=ifelse(Z[i]==s1-s2+1,1,0)
      mu[s1,s2]=sum(Z==s1-s2+1)/length(Z)
    }
  }
  re=D-mu
  return(kronecker(re,l))
})


m=5
samplesplit=function(m,Z){
  I=length(Z)
  J=max(Z)
  ssp=rep(0,I)
  for(j in 1:J){
    l=which(Z==j)
    s=sample(rep(1:m, length.out = length(l)))
    ssp[l]=s
  }
  return(ssp)
}
samples=samplesplit(m,Z)
dataf$sp=1
dataf$sp <- samples[ match(dataf$cluster, clusteru) ]
dataf$g=1
for(i in 1:m){
  for(j in 1:J){
    train=which(dataf$sp!=i&dataf$time==j)
    Ytrain=dataf$y[train]
    Xtrain=dataf[train,5:8]
    test=which(dataf$sp==i&dataf$time==j)
    Ytest= dataf$y[test]
    Xtest=dataf[test,5:8]
    
    traindata=data.frame(y=Ytrain,Xtrain)
    
    #unadjusted
    model=lm(y~1,data=traindata)
    ypred=predict(model,newdata = Xtest)
    
    #linear regression
    model=lm(y~.,data=traindata)
    ypred=predict(model,newdata = Xtest)
    
    
    #superlearner 
    learners <- c("SL.glm", "SL.randomForest", "SL.rpart", "SL.svm")
    SL_model <- SuperLearner(Y = Ytrain,X = Xtrain,family = gaussian(),
                             SL.library = learners,
                             method = "method.NNLS")
    ypred <- predict(SL_model, newdata = Xtest)$pred
    dataf$g[test]=ypred
  }
}

dataf$residual=dataf$y-dataf$g

re=dataf$residual




resi_listf=split(re,dataf$cluster)
Dsquare=Reduce("+",Map(function(a, b) t(a) %*% b, Dmu, Dmu)) 
Dsinv=solve(Dsquare)
Dvsresidual=Reduce("+",Map(function(a, b) t(a) %*% b, Dmu, resi_listf)) 
betaest=Dsinv%*%Dvsresidual
phi=Map(function(a, b) {t(a) %*% (b-a%*%betaest)}, Dmu, resi_listf)
meat=Reduce("+",Map(function(a, b) a %*% t(b), phi, phi)) 
vmatrix=Dsinv%*%meat%*%Dsinv

betaest1=betaest


num <- length(unique(dataf$time))

l1 <- diag(1, num)          
l2 <- matrix(1, num, num)   

Q1 <- rep(list(l1), I)
Q2 <- rep(list(l2), I)
Qfunction=function(i,t=1){
  q=list(Q1[[i]],Q2[[i]])
  return(q)
}
for(sss in 1:2){
  phiiold=map(1:I, function(i) {
    Q=Qfunction(i)
    phii=lapply(Q, function(Qq){
      t(Dmu[[i]])%*%Qq%*%(resi_listf[[i]]-Dmu[[i]]%*%betaest)
    })
    phii=do.call(rbind,phii)
    return (phii)
  })
  
  CI=Reduce("+",Map(function(a, b) a %*% t(b), phiiold, phiiold))
  CIinverse=solve(CI)
  phibeta=map(1:I, function(i) {
    Q=Qfunction(i)
    phii=lapply(Q, function(Qq){
      t(Dmu[[i]])%*%Qq%*%Dmu[[i]]
    })
    phii=do.call(rbind,phii)
    return (phii)
  })
  phibeta=Reduce("+",phibeta)
  phivarinverse=t(phibeta)%*%CIinverse%*%phibeta
  phivar=solve(phivarinverse)
  phis=map(1:I, function(i) {
    Q=Qfunction(i)
    phii=lapply(Q, function(Qq){
      t(Dmu[[i]])%*%Qq%*%resi_listf[[i]]
    })
    phii=do.call(rbind,phii)
    return (phii)
  })
  phis=Reduce("+",phis)
  betaest=phivar%*%t(phibeta)%*%CIinverse%*%phis
  betaest}






b1=mean(betaest1)
b2=mean(betaest)
l=J
sss=rep(1/l,l)
v1=t(sss)%*%vmatrix%*%sss
v2=t(sss)%*%phivar%*%sss


#period
library(haven)
library(dplyr)
library(np)
library(ggplot2)
library(tidyverse)
library(SuperLearner)
library(randomForest)
library(xgboost)
library(e1071)
library(rpart)
library(nnet)
library(Matrix)
library(glmnet)

data=read.csv()
data=data[order(data$id),]
data=data[order(data$time),]
data=data[order(data$cluster),]

d <- data[!duplicated(data[c("trained", "cluster")]),]
Z=d$trained
dataf=data[,-5]
dataf=dataf[,-5]
I=length(Z)
Ni=rep(1,I)
J=max(dataf$time)-1
clusteru=unique(dataf$cluster)
dataf=dataf[dataf$time%in%c(1:J),]

id=unique(dataf$id)
Dmu<-map(1:I, function(i){
  l=rep(1,Ni[i])
  D=matrix(0,J,J)
  
  mu=matrix(0,J,J)
  cu=cumsum(pi)
  for(j in 1:J){
    D[j,j]=ifelse(Z[i]<=j,1,0)
    mu[j,j]=sum(Z<=j)/length(Z)
  }
  re=D-mu
  return(kronecker(re,l))
})


m=5
samplesplit=function(m,Z){
  I=length(Z)
  J=max(Z)
  ssp=rep(0,I)
  for(j in 1:J){
    l=which(Z==j)
    s=sample(rep(1:m, length.out = length(l)))
    ssp[l]=s
  }
  return(ssp)
}
samples=samplesplit(m,Z)
dataf$sp=1
dataf$sp <- samples[ match(dataf$cluster, clusteru) ]
dataf$g=1
for(i in 1:m){
  for(j in 1:J){
    train=which(dataf$sp!=i&dataf$time==j)
    Ytrain=dataf$y[train]
    Xtrain=dataf[train,5:8]
    test=which(dataf$sp==i&dataf$time==j)
    Ytest= dataf$y[test]
    Xtest=dataf[test,5:8]
    
    traindata=data.frame(y=Ytrain,Xtrain)
    
    #unadjusted
    model=lm(y~1,data=traindata)
    ypred=predict(model,newdata = Xtest)
    
    #linear regression
    model=lm(y~.,data=traindata)
    ypred=predict(model,newdata = Xtest)
    
    
    #superlearner 
    learners <- c("SL.glm", "SL.randomForest", "SL.rpart", "SL.svm")
    SL_model <- SuperLearner(Y = Ytrain,X = Xtrain,family = gaussian(),
                             SL.library = learners,
                             method = "method.NNLS")
    ypred <- predict(SL_model, newdata = Xtest)$pred
    dataf$g[test]=ypred
  }
}

dataf$residual=dataf$y-dataf$g

re=dataf$residual




resi_listf=split(re,dataf$cluster)
Dsquare=Reduce("+",Map(function(a, b) t(a) %*% b, Dmu, Dmu)) 
Dsinv=solve(Dsquare)
Dvsresidual=Reduce("+",Map(function(a, b) t(a) %*% b, Dmu, resi_listf)) 
betaest=Dsinv%*%Dvsresidual
phi=Map(function(a, b) {t(a) %*% (b-a%*%betaest)}, Dmu, resi_listf)
meat=Reduce("+",Map(function(a, b) a %*% t(b), phi, phi)) 
vmatrix=Dsinv%*%meat%*%Dsinv

betaest1=betaest



num <- length(unique(dataf$time))

l1 <- diag(1, num)          
l2 <- matrix(1, num, num)   

Q1 <- rep(list(l1), I)
Q2 <- rep(list(l2), I)
Qfunction=function(i,t=1){
  q=list(Q1[[i]],Q2[[i]])
  return(q)
}
for(sss in 1:2){
  phiiold=map(1:I, function(i) {
    Q=Qfunction(i)
    phii=lapply(Q, function(Qq){
      t(Dmu[[i]])%*%Qq%*%(resi_listf[[i]]-Dmu[[i]]%*%betaest)
    })
    phii=do.call(rbind,phii)
    return (phii)
  })
  
  CI=Reduce("+",Map(function(a, b) a %*% t(b), phiiold, phiiold))
  CIinverse=solve(CI)
  phibeta=map(1:I, function(i) {
    Q=Qfunction(i)
    phii=lapply(Q, function(Qq){
      t(Dmu[[i]])%*%Qq%*%Dmu[[i]]
    })
    phii=do.call(rbind,phii)
    return (phii)
  })
  phibeta=Reduce("+",phibeta)
  phivarinverse=t(phibeta)%*%CIinverse%*%phibeta
  phivar=solve(phivarinverse)
  phis=map(1:I, function(i) {
    Q=Qfunction(i)
    phii=lapply(Q, function(Qq){
      t(Dmu[[i]])%*%Qq%*%resi_listf[[i]]
    })
    phii=do.call(rbind,phii)
    return (phii)
  })
  phis=Reduce("+",phis)
  betaest=phivar%*%t(phibeta)%*%CIinverse%*%phis
  betaest}







#saturated
library(haven)
library(dplyr)
library(np)
library(ggplot2)
library(tidyverse)
library(SuperLearner)
library(randomForest)
library(xgboost)
library(e1071)
library(rpart)
library(nnet)
library(Matrix)
library(glmnet)
library(MASS)


data=read.csv()
data=data[order(data$id),]
data=data[order(data$time),]
data=data[order(data$cluster),]

d <- data[!duplicated(data[c("trained", "cluster")]),]
Z=d$trained
dataf=data[,-5]
dataf=dataf[,-5]
I=length(Z)
Ni=rep(1,I)
J=max(dataf$time)-1
clusteru=unique(dataf$cluster)
dataf=dataf[dataf$time%in%c(1:J),]

id=unique(dataf$id)
Dmu<-map(1:I, function(i){
  l=rep(1,Ni[i])
  D=matrix(0,J,J*(J+1)/2)
  mu=matrix(0,J,J*(J+1)/2)
  tab=table(factor(Z,levels = 1:J))
  pvec=as.numeric(tab)/length(Z)
  cs=1
  for(j in 1:J){
    Hmu=rev(pvec[1:j])
    mu[j,cs:(cs+j-1)]=Hmu
    Hd=as.integer(Z[i]==j:1)
    D[j,cs:(cs+j-1)]=Hd
    cs=cs+j
  }
  re=D-mu
  return(kronecker(re,l))
})

m=5
samplesplit=function(m,Z){
  I=length(Z)
  J=max(Z)
  ssp=rep(0,I)
  for(j in 1:J){
    l=which(Z==j)
    s=sample(rep(1:m, length.out = length(l)))
    ssp[l]=s
  }
  return(ssp)
}
samples=samplesplit(m,Z)
dataf$sp=1
dataf$sp <- samples[ match(dataf$cluster, clusteru) ]
dataf$g=1
for(i in 1:m){
  for(j in 1:J){
    train=which(dataf$sp!=i&dataf$time==j)
    Ytrain=dataf$y[train]
    Xtrain=dataf[train,5:8]
    test=which(dataf$sp==i&dataf$time==j)
    Ytest= dataf$y[test]
    Xtest=dataf[test,5:8]
    
    traindata=data.frame(y=Ytrain,Xtrain)
    
    #unadjusted
    model=lm(y~1,data=traindata)
    ypred=predict(model,newdata = Xtest)
    
    #linear regression
    model=lm(y~.,data=traindata)
    ypred=predict(model,newdata = Xtest)
    
    
    #superlearner 
    learners <- c("SL.glm", "SL.randomForest", "SL.rpart", "SL.svm")
    SL_model <- SuperLearner(Y = Ytrain,X = Xtrain,family = gaussian(),
                             SL.library = learners,
                             method = "method.NNLS")
    ypred <- predict(SL_model, newdata = Xtest)$pred
    dataf$g[test]=ypred
  }
}

dataf$residual=dataf$y-dataf$g

re=dataf$residual




resi_listf=split(re,dataf$cluster)
re=NULL
Dsquare <- matrix(0, ncol(Dmu[[1]]), ncol(Dmu[[1]]))
for (i in seq_along(Dmu)) {
  Dsquare <- Dsquare + crossprod(Dmu[[i]])
}
Dsinv=ginv(Dsquare)
Dvsresidual <- 0
for (i in seq_along(Dmu)) {
  Dvsresidual <- Dvsresidual + t(Dmu[[i]]) %*% resi_listf[[i]]
} 
gc()
betaest=Dsinv%*%Dvsresidual
phi=Map(function(a, b) {t(a) %*% (b-a%*%betaest)}, Dmu, resi_listf)
meat <- 0
for (i in seq_along(phi)) {
  meat <- meat + phi[[i]] %*% t(phi[[i]])
}
vmatrix=Dsinv%*%meat%*%Dsinv

betaest1=betaest
b1=mean(betaest1)
b1
l=J*(J+1)/2
sss=rep(1/l,l)
v1=t(sss)%*%vmatrix%*%sss
sqrt(v1)

gc()

num <- length(unique(dataf$time))

l1 <- diag(1, num)          
l2 <- matrix(1, num, num)   

Q1 <- rep(list(l1), I)
Q2 <- rep(list(l2), I)
Qfunction=function(i,t=1){
  q=list(Q1[[i]],Q2[[i]])
  return(q)
}
for(sss in 1:2){
  phiiold=map(1:I, function(i) {
    Q=Qfunction(i)
    phii=lapply(Q, function(Qq){
      t(Dmu[[i]])%*%Qq%*%(resi_listf[[i]]-Dmu[[i]]%*%betaest)
    })
    phii=do.call(rbind,phii)
    return (phii)
  })
  
  CI <- 0
  for (i in seq_along(phiiold)) {
    CI <- CI + phiiold[[i]] %*% t(phiiold[[i]])
  }
  CIinverse=ginv(CI)
  CI=0
  gc()
  p <- ncol(Dmu[[1]])
  k <- 1
  phibeta_sum <- matrix(0, nrow = k * p, ncol = p)
  
  for (i in 1:I) {
    Qs <- Qfunction(i)  
    
    for (j in seq_along(Qs)) {
      Qq <- Qs[[j]]
      
      tmp <- Qq %*% Dmu[[i]]
      contrib <- crossprod(Dmu[[i]], tmp)  # p x p
      
      rows <- ((j - 1) * p + 1):(j * p)
      phibeta_sum[rows, ] <- phibeta_sum[rows, ] + contrib
    }
  }
  contrib=0
  gc()
  phibeta=phibeta_sum
  phivarinverse=t(phibeta)%*%CIinverse%*%phibeta
  phivar=ginv(phivarinverse)
  phis=map(1:I, function(i) {
    Q=Qfunction(i)
    phii=lapply(Q, function(Qq){
      t(Dmu[[i]])%*%Qq%*%resi_listf[[i]]
    })
    phii=do.call(rbind,phii)
    return (phii)
  })
  phis_sum <- phis[[1]]
  for (i in 2:length(phis)) {
    phis_sum <- phis_sum + phis[[i]]
  }
  phis=phis_sum
  gc()
  betaest=phivar%*%t(phibeta)%*%CIinverse%*%phis
  betaest}
betaest
phivar

b1=mean(betaest1)
b2=mean(betaest)
l=J*(J+1)/2
sss=rep(1/l,l)
v1=t(sss)%*%vmatrix%*%sss
v2=t(sss)%*%phivar%*%sss

