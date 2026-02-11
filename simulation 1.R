# ------------
#simulation for the constant treatment effect.
#cluster and individual randomized trial. 
#method: unadjusted, linear and superlearner.
#variance: independence structure vs QIF adjustment.
# ------------

#I=20 J=3
#For cluster, I can be 100, J can be 5.
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
I=20
m=2
J=3 
Jinf=J-1 
pi=rep(1/J,J)

Ni=rep(20,I) 
#split sample
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
#generate D matrix
Dgenerate=function(setting,z,j){
  if(setting=="constant"|setting=="period"){
    return(ifelse(z<=j,1,0))
  }
  if(setting=="duration"|setting=="saturated"){
    Dij=0
    for(d in 1:J){
      if(d==j-z+1){
        Dij=d
      }
    }
    return(Dij)
  }
}
#choose Q
Qfunction=function(i,t=1){
  q=list(Q1[[i]],Q2[[i]],Q3[[i]])
  return(q)
}

Nij=matrix(0,I,J)
for(i in 1:I){
  for(j in 1:J){
    Nij[i,j]=sample(5:15, 1)
  }
}
while (TRUE) {
  Z= sample(1:J, size = I, replace = TRUE,prob = pi)
  l=0
  for(j in 1:J){
    if(sum(Z==j)>=m){
      l=l+1
    }
  }
  if(l==J){
    break
  }
} 
#generate X and Y
X <- map(1:I, function(i) {
  X1 <- rnorm(1,0,1)
  X2 <- rbinom(Ni[i], size = 1, prob = 0.5)
  X3 <- rnorm(Ni[i],0,1)
  X4 <- rnorm(Ni[i],0,1)
  data.frame(cbind(X1,X2,X3,X4))
})

Y<-map(1:I,function(i){
  YA=matrix(0,Ni[i],J)
  time_error=rnorm(Ni[i],0,sqrt(0.1))
  cluster <- rnorm(1,0,sqrt(0.1))
  for(j in 1:J){
    D=Dgenerate("constant",Z[i],j)
    treatment=D*(1+X[[i]]$X3-mean(X[[i]]$X3)+(X[[i]]$X4^3-mean(X[[i]]$X4^3))/2)
    time_specific=rnorm(1,0,sqrt(0.1))
    YA[,j]=exp(X[[i]]$X1*X[[i]]$X2)+2*(X[[i]]$X3>1)+(X[[i]]$X4>-1)+(X[[i]]$X1>0.5)*(j+1)+X[[i]]$X4^2/2+treatment
    +rnorm(Ni[i],0,sqrt(0.7))+cluster+time_error+time_specific
  }
  return(YA)
})
#generate D-mu
samples=samplesplit(m,Z)
Dmu<-map(1:I, function(i){
  l=rep(1,Ni[i])
  D=rep(0,J)
  
  mu=rep(0,J)
  for(j in 1:J){
    D[j]=ifelse(Z[i]<=j,1,0)
    mu[j]=sum(Z<=j)/length(Z)
  }
  # mu=cumsum(pi)
  re=D-mu
  return(kronecker(re,l))
})
D=do.call(c,Dmu)
X=do.call(rbind,X)
Y=do.call(rbind,Y)
Y=as.data.frame(Y)
colnames(Y)=c("Y1","Y2","Y3")
fdata=data.frame(id=1:(I*Ni[1]),cluster=rep(1:I,each=Ni[1]),sp=rep(samples,each=Ni[1]),Y,X)
df <- fdata %>%
  pivot_longer(
    cols = starts_with("Y"),         
    names_to = "time",
    names_prefix = "Y",
    values_to = "Y"
  ) %>%
  mutate(
    time = as.integer(time),        
  )
df=df[order(df$time),]
df=df[order(df$cluster),]
df$D=D 
#get observed data
obserl=NULL
for(i in 1:I){
  for(j in 1:J){
    idc=df$id[df$cluster==i&df$time==j]
    idc=sample(idc,Nij[i,j])
    obser=data.frame(id=idc,time=j)
    obserl=rbind(obserl,obser)
  }
}
df$fid=1:length(df$id)
observed=inner_join(df,obserl,by=c("id","time"))
observed$g=1
#regression
for(i in 1:m){
  for(j in 1:J){
    train=which(observed$sp!=i&observed$time==j)
    Ytrain=observed$Y[train]
    Xtrain=observed[train,4:7]
    test=which(observed$sp==i&observed$time==j)
    Ytest= observed$Y[test]
    Xtest=observed[test,4:7]
    
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
    
    observed$g[test]=ypred
  }
}
observed$residual=observed$Y-observed$g
for(i in 1:I){
  for(j in 1:J){
    l=which(observed$cluster==i&observed$time==j)
    observed$D[l]=observed$D[l]/sqrt(Nij[i,j])
    observed$residual[l]=observed$residual[l]/sqrt(Nij[i,j])
  }
}
Dmu=observed$D
re=observed$residual
#basic results
Dmu=split(Dmu,observed$cluster)
resi_listf=split(re,observed$cluster)
Dsquare=Reduce("+",Map(function(a, b) t(a) %*% b, Dmu, Dmu)) 
Dsinv=solve(Dsquare)
Dvsresidual=Reduce("+",Map(function(a, b) t(a) %*% b, Dmu, resi_listf)) 
betaest=Dsinv%*%Dvsresidual
phi=Map(function(a, b) {t(a) %*% (b-a%*%betaest)}, Dmu, resi_listf)
meat=Reduce("+",Map(function(a, b) a %*% t(b), phi, phi)) 
vmatrix=Dsinv%*%meat%*%Dsinv

betaes=betaest
vm=vmatrix

#QIF
#Q function
idlist=observed$fid
idlist=idlist%%(Ni[1]*J)
idlist[idlist==0]=Ni[1]*J
idlist=split(idlist,observed$cluster)
Q1=list()
Q2=list()
Q3=list()
Q4=list()
dime=Ni*J
for(i in 1:I){
  num=length(idlist[[i]])
  l=matrix(0,num,num)
  diag(l)=1
  Q1[[i]]=l
  Q2[[i]]=matrix(1,num,num)
  block_id=(idlist[[i]] - 1) %/% Ni[i]
  Q3[[i]]= outer(block_id, block_id, "==") * 1
  mod_vals=(idlist[[i]]) %% Ni[i]
  Q4[[i]]=outer(mod_vals, mod_vals, "==")*1
}
for(sss in 1:10){
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

CI=Reduce("+",Map(function(a, b) a %*% t(b), phiiold, phiiold))
CIinverse=ginv(CI)
phibeta=map(1:I, function(i) {
  Q=Qfunction(i)
  phii=lapply(Q, function(Qq){
    t(Dmu[[i]])%*%Qq%*%Dmu[[i]]
  })
  phii=do.call(rbind,phii)
  return (phii)
})
cplus1=map(1:I, function(i) {
  phii=phibeta[[i]][,1]%*%t(phiiold[[i]])+phiiold[[i]]%*%t(phibeta[[i]][,1])
  return (phii)
})
#variance adjustment
phibeta=Reduce("+",phibeta)
cplus1=Reduce("+",cplus1)
phiiold=Reduce("+",phiiold)
phivarinverse=t(phibeta)%*%CIinverse%*%phibeta
phivar=ginv(phivarinverse)
G1=phivar%*%t(phibeta)%*%CIinverse%*%cplus1%*%CIinverse%*%phiiold
G=G1
fv1=(diag(1,nrow(G))+G)%*%phivar%*%t(diag(1,nrow(G))+G)
SNL=(diag(1,nrow(G))+G)%*%phivar%*%t(phibeta)%*%CIinverse
Oi=map(1:I, function(i) {
  Q=Qfunction(i)
  o=lapply(Q, function(Qq){
    t(Dmu[[i]])%*%Qq
  })
  oi=Dmu[[i]]%*%SNL%*%do.call(rbind,o)
  return (oi)
})
phiinew=map(1:I, function(i) {
  Q=Qfunction(i)
  phii=lapply(Q, function(Qq){
    t(Dmu[[i]])%*%Qq%*%ginv(-Oi[[i]]+diag(1,nrow(Oi[[i]])))%*%(resi_listf[[i]]-Dmu[[i]]%*%betaest)
  })
  phii=do.call(rbind,phii)
  return (phii)
})
CInew=Reduce("+",Map(function(a, b) a %*% t(b), phiinew, phiinew))
fv=(diag(1,nrow(G))+G)%*%phivar%*%t(phibeta)%*%CIinverse%*%CInew%*%CIinverse%*%
  phibeta%*%phivar%*%t(diag(1,nrow(G))+G)





betavif=betaest
vmvif=fv


da=data.frame(betabase=betaes,sdbase=sqrt(vm),betavif=betavif,sdvif=sqrt(vmvif))


#I=1000 J=20 individual
#same structure as above
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
I=1000
m=5
J=20
Jinf=J-1 
pi=rep(1/J,J)

Ni=rep(1,I) 
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
Dgenerate=function(setting,z,j){
  if(setting=="constant"|setting=="period"){
    return(ifelse(z<=j,1,0))
  }
  if(setting=="duration"|setting=="saturated"){
    Dij=0
    for(d in 1:J){
      if(d==j-z+1){
        Dij=d
      }
    }
    return(Dij)
  }
}
#Qfunction
Qfunction=function(i,t=1){
  q=list(Q1[[i]],Q2[[i]])
  return(q)
}

Nij=matrix(0,I,J)
for(i in 1:I){
  for(j in 1:J){
    Nij[i,j]=sample(0:1, 1)
  }
  while(sum(Nij[i,])==0){
    for(j in 1:J){
      Nij[i,j]=sample(0:1, 1)
    }
  }
}
while (TRUE) {
  Z= sample(1:J, size = I, replace = TRUE,prob = pi)
  l=0
  for(j in 1:J){
    if(sum(Z==j)>=m){
      l=l+1
    }
  }
  if(l==J){
    break
  }
} 
#generate data
X <- map(1:I, function(i) {
  X1 <- rnorm(1,0,1)
  X2 <- rbinom(Ni[i], size = 1, prob = 0.5)
  X3 <- rnorm(Ni[i],0,1)
  X4 <- rnorm(Ni[i],0,1)
  data.frame(cbind(X1,X2,X3,X4))
})

Y<-map(1:I,function(i){
  YA=matrix(0,Ni[i],J)
  time_error=rnorm(Ni[i],0,sqrt(0.1))
  cluster <- rnorm(1,0,sqrt(0.1))
  for(j in 1:J){
    D=Dgenerate("constant",Z[i],j)
    treatment=D*(1+X[[i]]$X3-mean(X[[i]]$X3)+(X[[i]]$X4^3-mean(X[[i]]$X4^3))/2)
    time_specific=rnorm(1,0,sqrt(0.1))
    YA[,j]=exp(X[[i]]$X1*X[[i]]$X2)+2*(X[[i]]$X3>1)+(X[[i]]$X4>-1)+(X[[i]]$X1>0.5)*(j+1)+X[[i]]$X4^2/2+treatment
    +rnorm(Ni[i],0,sqrt(0.9))+time_error
  }
  return(YA)
})

samples=samplesplit(m,Z)
#D matrix
Dmu<-map(1:I, function(i){
  l=rep(1,Ni[i])
  D=rep(0,J)
  
  mu=rep(0,J)
  for(j in 1:J){
    D[j]=ifelse(Z[i]<=j,1,0)
    mu[j]=sum(Z<=j)/length(Z)
  }
  # mu=cumsum(pi)
  re=D-mu
  return(kronecker(re,l))
})
D=do.call(c,Dmu)
X=do.call(rbind,X)
Y=do.call(rbind,Y)
Y=as.data.frame(Y)
colnames(Y)=paste0("Y",1:J)
fdata=data.frame(id=1:(I*Ni[1]),cluster=rep(1:I,each=Ni[1]),sp=rep(samples,each=Ni[1]),Y,X)
df <- fdata %>%
  pivot_longer(
    cols = starts_with("Y"),         
    names_to = "time",
    names_prefix = "Y",
    values_to = "Y"
  ) %>%
  mutate(
    time = as.integer(time),        
  )
df=df[order(df$time),]
df=df[order(df$cluster),]
df$D=D 

obserl=NULL
for(i in 1:I){
  for(j in 1:J){
    idc=df$id[df$cluster==i&df$time==j]
    # idc=sample(idc,Nij[i,j])
    if(Nij[i,j]>0){
      obser=data.frame(id=idc,time=j)
      obserl=rbind(obserl,obser)
    }
  }
}
df$fid=1:length(df$id)
observed=inner_join(df,obserl,by=c("id","time"))
observed$g=1
for(i in 1:m){
  for(j in 1:J){
    train=which(observed$sp!=i&observed$time==j)
    Ytrain=observed$Y[train]
    Xtrain=observed[train,4:7]
    test=which(observed$sp==i&observed$time==j)
    Ytest= observed$Y[test]
    Xtest=observed[test,4:7]
    
    
    traindata=data.frame(y=Ytrain,Xtrain)
    
    #Unadjusted 
    model=lm(y~1,data=traindata)
    ypred=predict(model,newdata = Xtest)
    
    #linear regression
    model=lm(y~.,data=traindata)
    ypred=predict(model,newdata = Xtest)
    
    #superlearner crossvalidation
    learners <- c("SL.glm", "SL.randomForest", "SL.rpart", "SL.svm")
    SL_model <- SuperLearner(Y = Ytrain,X = Xtrain,family = gaussian(),
                             SL.library = learners,
                             method = "method.NNLS")
    ypred <- predict(SL_model, newdata = Xtest)$pred
    
    observed$g[test]=ypred
  }
}
observed$residual=observed$Y-observed$g

Dmu=observed$D
re=observed$residual

Dmu=split(Dmu,observed$cluster)
resi_listf=split(re,observed$cluster)
Dsquare=Reduce("+",Map(function(a, b) t(a) %*% b, Dmu, Dmu)) 
Dsinv=solve(Dsquare)
Dvsresidual=Reduce("+",Map(function(a, b) t(a) %*% b, Dmu, resi_listf)) 
betaest=Dsinv%*%Dvsresidual
phi=Map(function(a, b) {t(a) %*% (b-a%*%betaest)}, Dmu, resi_listf)
meat=Reduce("+",Map(function(a, b) a %*% t(b), phi, phi)) 
vmatrix=Dsinv%*%meat%*%Dsinv

betaes=betaest
vm=vmatrix


#Q function
idlist=observed$fid
idlist=idlist%%(Ni[1]*J)
idlist[idlist==0]=Ni[1]*J
idlist=split(idlist,observed$cluster)
Q1=list()
Q2=list()
Q3=list()
Q4=list()
dime=Ni*J
for(i in 1:I){
  num=length(idlist[[i]])
  l=matrix(0,num,num)
  diag(l)=1
  Q1[[i]]=l
  Q2[[i]]=matrix(1,num,num)
  block_id=(idlist[[i]] - 1) %/% Ni[i]
  Q3[[i]]= outer(block_id, block_id, "==") * 1
  mod_vals=(idlist[[i]]) %% Ni[i]
  Q4[[i]]=outer(mod_vals, mod_vals, "==")*1
}
for(sss in 1:10){
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

CI=Reduce("+",Map(function(a, b) a %*% t(b), phiiold, phiiold))
CIinverse=ginv(CI)
phibeta=map(1:I, function(i) {
  Q=Qfunction(i)
  phii=lapply(Q, function(Qq){
    t(Dmu[[i]])%*%Qq%*%Dmu[[i]]
  })
  phii=do.call(rbind,phii)
  return (phii)
})
cplus1=map(1:I, function(i) {
  phii=phibeta[[i]][,1]%*%t(phiiold[[i]])+phiiold[[i]]%*%t(phibeta[[i]][,1])
  return (phii)
})
#variance adjustment
phibeta=Reduce("+",phibeta)
cplus1=Reduce("+",cplus1)
phiiold=Reduce("+",phiiold)
phivarinverse=t(phibeta)%*%CIinverse%*%phibeta
phivar=ginv(phivarinverse)
G1=phivar%*%t(phibeta)%*%CIinverse%*%cplus1%*%CIinverse%*%phiiold
G=G1
fv1=(diag(1,nrow(G))+G)%*%phivar%*%t(diag(1,nrow(G))+G)
SNL=(diag(1,nrow(G))+G)%*%phivar%*%t(phibeta)%*%CIinverse
Oi=map(1:I, function(i) {
  Q=Qfunction(i)
  o=lapply(Q, function(Qq){
    t(Dmu[[i]])%*%Qq
  })
  oi=Dmu[[i]]%*%SNL%*%do.call(rbind,o)
  return (oi)
})
phiinew=map(1:I, function(i) {
  Q=Qfunction(i)
  phii=lapply(Q, function(Qq){
    t(Dmu[[i]])%*%Qq%*%ginv(-Oi[[i]]+diag(1,nrow(Oi[[i]])))%*%(resi_listf[[i]]-Dmu[[i]]%*%betaest)
  })
  phii=do.call(rbind,phii)
  return (phii)
})
CInew=Reduce("+",Map(function(a, b) a %*% t(b), phiinew, phiinew))
fv=(diag(1,nrow(G))+G)%*%phivar%*%t(phibeta)%*%CIinverse%*%CInew%*%CIinverse%*%
  phibeta%*%phivar%*%t(diag(1,nrow(G))+G)





betavif=betaest
vmvif=fv


da=data.frame(betabase=betaes,sdbase=sqrt(vm),betavif=betavif,sdvif=sqrt(vmvif))
