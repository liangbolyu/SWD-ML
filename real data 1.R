
#constant treatment effect
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
datab=read.csv("")#IPOS data
J=5
data=data[order(data$TInStudy),]
data=data[order(data$person_id),]
table(data$person_id)
t=table(data$person_id)
l=which(t>1)

ch=names(l)
data=data[data$person_id%in%ch,]

data$time=rep(1,length(data$cluster_group_id))
data[data$TInStudy<=50,]$time=0
data[(data$TInStudy>50)&(data$TInStudy<=140),]$time=1
data[(data$TInStudy>140)&(data$TInStudy<=230),]$time=2
data[(data$TInStudy>230)&(data$TInStudy<=320),]$time=3
data[(data$TInStudy>320)&(data$TInStudy<=400),]$time=4
data[data$TInStudy>400,]$time=5

s=duplicated(data$person_id)
l=which(s==FALSE)
l
database=data[l,]
l=which(data$time>0)
datal=data[l,]

length(unique(database$person_id))
length(unique(datal$person_id))

database=database[database$person_id%in%unique(datal$person_id),]
length(unique(database$person_id))


X=data.frame(id=database$person_id,x1=database$death,x2=database$sex,x3=database$age,x4=database$Total)
Y=data.frame(cp=datal$cluster_group_id,cluster=datal$cluster_id,id=datal$person_id,time=datal$time,y=datal$Total)

datau=Y%>%left_join(X,by="id")
datau=datau[order(datau$time),]
datau=datau[order(datau$cluster),]

database=database[order(database$time),]
database=database[order(database$cluster_id),]


dataf=NULL
id=unique(database$person_id)
S=rep(1,length(id)*J)
t=1

for(i in 1:length(id)){
  for(j in 1:J){
    l=which((datau$id==id[i])&(datau$time==j))
    if(length(l)==1){
      dataf=rbind(dataf,datau[l,])
    }
    if(length(l)>1){
      d=datau[l[1],]
      d$y=mean(datau$y[l])
      dataf=rbind(dataf,d)
    }
  }
}

dataf=dataf[order(dataf$id),]
dataf=dataf[order(dataf$time),]
dataf=dataf[order(dataf$cluster),]
S=rep(1,length(id)*J)
t=1
for(ss in 1:length(unique(dataf$cluster))){
  idc=unique(dataf[dataf$cluster==unique(dataf$cluster)[ss],]$id)
  for(j in 1:J){
    for(i in 1:length(idc)){
      l=which((dataf$id==idc[i])&(dataf$time==j))
      if(length(l)==0){
        S[t]=0
      }
      t=t+1
    }
  }
}
l=(!duplicated(dataf$cluster))
Z=dataf$cp[l]

datab=datab[datab$person_id%in%dataf$id,]
iid=unique(datab$person_id)
dataf$x5=1
dataf$x6=1
for(i in 1:length(iid)){
  l=which(datab$person_id==iid[i])
  d=datab[l,]
  d=d[order(d$TInStudy),]
  l=which(dataf$id==iid[i])
  dataf[l,]$x5=d$DPII[1]
  dataf[l,]$x6=d$DEBI[1]
}

Ni=rep(0,length(unique(dataf$cluster)))
clusteru=unique(dataf$cluster)


Nij=matrix(0,length(unique(dataf$cluster)),J)
for(i in 1:length(unique(dataf$cluster))){
  l=dataf[dataf$cluster==clusteru[i],]
  Ni[i]=length(unique(l$id))
  for(j in 1:J){
    l=which((dataf$cluster==clusteru[i])&(dataf$time==j))
    Nij[i,j]=length(l)
  }
}
I=length(clusteru)
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
m=2
samples=samplesplit(m,Z)
dataf$sp=1
for(i in 1:I){
  dataf[dataf$cluster==clusteru[i],]$sp=samples[i]
}
dataf$g=1
for(i in 1:m){
  for(j in 1:J){
    train=which(dataf$sp!=i&dataf$time==j)
    Ytrain=dataf$y[train]
    Xtrain=dataf[train,7:11]
    test=which(dataf$sp==i&dataf$time==j)
    Ytest= dataf$y[test]
    Xtest=dataf[test,7:11]

    
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
D=do.call(c,Dmu)
dataf$residual=dataf$y-dataf$g
l=which(S==1)
D=D[l]
for(i in 1:I){
  for(j in 1:J){
    l=which(dataf$cluster==clusteru[i]&dataf$time==j)
    if(length(l)!=0){
      D[l]=D[l]/sqrt(Nij[i,j])
      dataf$residual[l]=dataf$residual[l]/sqrt(Nij[i,j])
    }
  }
}


Dmu=D
re=dataf$residual



Dmu=split(Dmu,dataf$cluster)
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

b1=betaest
v1=vmatrix

idlist=which(S==1)
idlist=split(idlist,dataf$cluster)
for(i in 1:I){
  idlist[[i]]=idlist[[i]]-idlist[[i]][1]+1
  idlist[[i]]=idlist[[i]]%%(Ni[i]*J)
  idlist[[i]][idlist[[i]]==0]=Ni[i]*J
}
betaest=betaest1
Q1=list()
Q2=list()
Q3=list()
Q4=list()
for(i in 1:I){
  num=length(idlist[[i]])
  l=matrix(0,num,num)
  diag(l)=1
  Q1[[i]]=l
  l2 <- matrix(1,num,num)
  Q2[[i]]=l2
  
  block_id=(idlist[[i]] - 1) %/% Ni[i]
  l3= outer(block_id, block_id, "==") * 1
  Q3[[i]]=l3
  mod_vals=(idlist[[i]]) %% Ni[i]
  Q4[[i]]=outer(mod_vals, mod_vals, "==")*1
}
Qfunction=function(i,t=1){
  q=list(Q1[[i]],Q2[[i]],Q3[[i]])
  return(q)
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
library(MASS)
  
  
data=read.csv()
datab=read.csv()
J=5
data=data[order(data$TInStudy),]
data=data[order(data$person_id),]
table(data$person_id)
t=table(data$person_id)
l=which(t>1)

ch=names(l)
data=data[data$person_id%in%ch,]

data$time=rep(1,length(data$cluster_group_id))
data[data$TInStudy<=50,]$time=0
data[(data$TInStudy>50)&(data$TInStudy<=140),]$time=1
data[(data$TInStudy>140)&(data$TInStudy<=230),]$time=2
data[(data$TInStudy>230)&(data$TInStudy<=320),]$time=3
data[(data$TInStudy>320)&(data$TInStudy<=400),]$time=4
data[data$TInStudy>400,]$time=5


s=duplicated(data$person_id)
l=which(s==FALSE)
l
database=data[l,]
l=which(data$time>0)
datal=data[l,]

length(unique(database$person_id))
length(unique(datal$person_id))

database=database[database$person_id%in%unique(datal$person_id),]
length(unique(database$person_id))


X=data.frame(id=database$person_id,x1=database$death,x2=database$sex,x3=database$age,x4=database$Total)
Y=data.frame(cp=datal$cluster_group_id,cluster=datal$cluster_id,id=datal$person_id,time=datal$time,y=datal$Total)

datau=Y%>%left_join(X,by="id")
datau=datau[order(datau$time),]
datau=datau[order(datau$cluster),]

database=database[order(database$time),]
database=database[order(database$cluster_id),]


dataf=NULL
id=unique(database$person_id)
S=rep(1,length(id)*J)
t=1

for(i in 1:length(id)){
  for(j in 1:J){
    l=which((datau$id==id[i])&(datau$time==j))
    if(length(l)==1){
      dataf=rbind(dataf,datau[l,])
    }
    if(length(l)>1){
      d=datau[l[1],]
      d$y=mean(datau$y[l])
      dataf=rbind(dataf,d)
    }
  }
}

dataf=dataf[order(dataf$id),]
dataf=dataf[order(dataf$time),]
dataf=dataf[order(dataf$cluster),]
S=rep(1,length(id)*J)
t=1
for(ss in 1:length(unique(dataf$cluster))){
  idc=unique(dataf[dataf$cluster==unique(dataf$cluster)[ss],]$id)
  for(j in 1:J){
    for(i in 1:length(idc)){
      l=which((dataf$id==idc[i])&(dataf$time==j))
      if(length(l)==0){
        S[t]=0
      }
      t=t+1
    }
  }
}
l=(!duplicated(dataf$cluster))
Z=dataf$cp[l]

datab=datab[datab$person_id%in%dataf$id,]
iid=unique(datab$person_id)
dataf$x5=1
dataf$x6=1
for(i in 1:length(iid)){
  l=which(datab$person_id==iid[i])
  d=datab[l,]
  d=d[order(d$TInStudy),]
  l=which(dataf$id==iid[i])
  dataf[l,]$x5=d$DPII[1]
  dataf[l,]$x6=d$DEBI[1]
}

Ni=rep(0,length(unique(dataf$cluster)))
clusteru=unique(dataf$cluster)


Nij=matrix(0,length(unique(dataf$cluster)),J)
for(i in 1:length(unique(dataf$cluster))){
  l=dataf[dataf$cluster==clusteru[i],]
  Ni[i]=length(unique(l$id))
  for(j in 1:J){
    l=which((dataf$cluster==clusteru[i])&(dataf$time==j))
    Nij[i,j]=length(l)
  }
}
I=length(clusteru)
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
m=2
samples=samplesplit(m,Z)
dataf$sp=1
for(i in 1:I){
  dataf[dataf$cluster==clusteru[i],]$sp=samples[i]
}
dataf$g=1
for(i in 1:m){
  for(j in 1:J){
    train=which(dataf$sp!=i&dataf$time==j)
    Ytrain=dataf$y[train]
    Xtrain=dataf[train,7:11]
    test=which(dataf$sp==i&dataf$time==j)
    Ytest= dataf$y[test]
    Xtest=dataf[test,7:11]
    
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
D=do.call(rbind,Dmu)
dataf$residual=dataf$y-dataf$g
l=which(S==1)
D=D[l,]
for(i in 1:I){
  for(j in 1:J){
    l=which(dataf$cluster==clusteru[i]&dataf$time==j)
    if(length(l)!=0){
      D[l,]=D[l,]/sqrt(Nij[i,j])
      dataf$residual[l]=dataf$residual[l]/sqrt(Nij[i,j])
    }
  }
}


Dmu=D
re=dataf$residual



Dmu_list <- split(seq_len(nrow(Dmu)), dataf$cluster)   
Dmu <- lapply(Dmu_list, function(idx) Dmu[idx, , drop = FALSE])
resi_listf=split(re,dataf$cluster)
Dsquare=Reduce("+",Map(function(a, b) t(a) %*% b, Dmu, Dmu)) 
Dsinv=solve(Dsquare)
Dvsresidual=Reduce("+",Map(function(a, b) t(a) %*% b, Dmu, resi_listf)) 
betaest=Dsinv%*%Dvsresidual
phi=Map(function(a, b) {t(a) %*% (b-a%*%betaest)}, Dmu, resi_listf)
meat=Reduce("+",Map(function(a, b) a %*% t(b), phi, phi)) 
vmatrix=Dsinv%*%meat%*%Dsinv

betaest
betaest1=betaest
vmatrix

idlist=which(S==1)
idlist=split(idlist,dataf$cluster)
for(i in 1:I){
  idlist[[i]]=idlist[[i]]-idlist[[i]][1]+1
  idlist[[i]]=idlist[[i]]%%(Ni[i]*J)
  idlist[[i]][idlist[[i]]==0]=Ni[i]*J
}
Q1=list()
Q2=list()
Q3=list()
Q4=list()
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
Qfunction=function(i,t=1){
  q=list(Q1[[i]],Q2[[i]],Q3[[i]],Q4[[i]])
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
betaest
phivar

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
cplus1=map(1:I, function(i) {
  phii=phibeta[[i]][,1]%*%t(phiiold[[i]])+phiiold[[i]]%*%t(phibeta[[i]][,1])
  return (phii)
})
cplus2=map(1:I, function(i) {
  phii=phibeta[[i]][,2]%*%t(phiiold[[i]])+phiiold[[i]]%*%t(phibeta[[i]][,2])
  return (phii)
})
cplus3=map(1:I, function(i) {
  phii=phibeta[[i]][,3]%*%t(phiiold[[i]])+phiiold[[i]]%*%t(phibeta[[i]][,3])
  return (phii)
})
cplus4=map(1:I, function(i) {
  phii=phibeta[[i]][,4]%*%t(phiiold[[i]])+phiiold[[i]]%*%t(phibeta[[i]][,4])
  return (phii)
})
cplus5=map(1:I, function(i) {
  phii=phibeta[[i]][,5]%*%t(phiiold[[i]])+phiiold[[i]]%*%t(phibeta[[i]][,5])
  return (phii)
})
phibeta=Reduce("+",phibeta)
cplus1=Reduce("+",cplus1)
cplus2=Reduce("+",cplus2)
cplus3=Reduce("+",cplus3)
cplus4=Reduce("+",cplus4)
cplus5=Reduce("+",cplus5)


phiiold=Reduce("+",phiiold)
phivarinverse=t(phibeta)%*%CIinverse%*%phibeta
phivar=solve(phivarinverse)
G1=phivar%*%t(phibeta)%*%CIinverse%*%cplus1%*%CIinverse%*%phiiold
G2=phivar%*%t(phibeta)%*%CIinverse%*%cplus2%*%CIinverse%*%phiiold
G3=phivar%*%t(phibeta)%*%CIinverse%*%cplus3%*%CIinverse%*%phiiold
G4=phivar%*%t(phibeta)%*%CIinverse%*%cplus4%*%CIinverse%*%phiiold
G5=phivar%*%t(phibeta)%*%CIinverse%*%cplus5%*%CIinverse%*%phiiold


G=cbind(G1,G2,G3,G4,G5)
fv1=(diag(1,nrow(G))+G)%*%phivar%*%t((diag(1,nrow(G))+G))
SNL=phivar%*%t(phibeta)%*%CIinverse
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
    t(Dmu[[i]])%*%Qq%*%solve(-Oi[[i]]+diag(1,nrow(Oi[[i]])))%*%(resi_listf[[i]]-Dmu[[i]]%*%betaest)
  })
  phii=do.call(rbind,phii)
  return (phii)
})
CInew=Reduce("+",Map(function(a, b) a %*% t(b), phiinew, phiinew))
fv=(diag(1,nrow(G))+G)%*%phivar%*%t(phibeta)%*%CIinverse%*%CInew%*%CIinverse%*%
  phibeta%*%phivar%*%t(diag(1,nrow(G))+G)

b1=mean(betaest1)
b2=mean(betaest)
sss=c(1/J,1/J,1/J,1/J,1/J)
v1=t(sss)%*%vmatrix%*%sss
v2=t(sss)%*%fv%*%sss







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
datab=read.csv()
J=3
data=data[order(data$TInStudy),]
data=data[order(data$person_id),]
table(data$person_id)
t=table(data$person_id)
l=which(t>1)

ch=names(l)
data=data[data$person_id%in%ch,]

data$time=rep(1,length(data$cluster_group_id))
data[data$TInStudy<=50,]$time=0
data[(data$TInStudy>50)&(data$TInStudy<=140),]$time=1
data[(data$TInStudy>140)&(data$TInStudy<=230),]$time=2
data[(data$TInStudy>230)&(data$TInStudy<=320),]$time=3
data[(data$TInStudy>320)&(data$TInStudy<=400),]$time=4
data[data$TInStudy>400,]$time=5

s=duplicated(data$person_id)
l=which(s==FALSE)
l
database=data[l,]
l=which(data$time>0)
datal=data[l,]

length(unique(database$person_id))
length(unique(datal$person_id))

database=database[database$person_id%in%unique(datal$person_id),]
length(unique(database$person_id))


X=data.frame(id=database$person_id,x1=database$death,x2=database$sex,x3=database$age,x4=database$Total)
Y=data.frame(cp=datal$cluster_group_id,cluster=datal$cluster_id,id=datal$person_id,time=datal$time,y=datal$Total)

datau=Y%>%left_join(X,by="id")
datau=datau[order(datau$time),]
datau=datau[order(datau$cluster),]

database=database[order(database$time),]
database=database[order(database$cluster_id),]


dataf=NULL
id=unique(database$person_id)
S=rep(1,length(id)*J)
t=1

for(i in 1:length(id)){
  for(j in 1:J){
    l=which((datau$id==id[i])&(datau$time==j))
    if(length(l)==1){
      dataf=rbind(dataf,datau[l,])
    }
    if(length(l)>1){
      d=datau[l[1],]
      d$y=mean(datau$y[l])
      dataf=rbind(dataf,d)
    }
  }
}

dataf=dataf[order(dataf$id),]
dataf=dataf[order(dataf$time),]
dataf=dataf[order(dataf$cluster),]
dataf=dataf[dataf$time%in%c(1,2),]
J=max(dataf$time)

S=rep(1,length(id)*J)
t=1
for(ss in 1:length(unique(dataf$cluster))){
  idc=unique(dataf[dataf$cluster==unique(dataf$cluster)[ss],]$id)
  for(j in 1:J){
    for(i in 1:length(idc)){
      l=which((dataf$id==idc[i])&(dataf$time==j))
      if(length(l)==0){
        S[t]=0
      }
      t=t+1
    }
  }
}
l=(!duplicated(dataf$cluster))
Z=dataf$cp[l]

datab=datab[datab$person_id%in%dataf$id,]
iid=unique(datab$person_id)
dataf$x5=1
dataf$x6=1
for(i in 1:length(iid)){
  l=which(datab$person_id==iid[i])
  d=datab[l,]
  d=d[order(d$TInStudy),]
  l=which(dataf$id==iid[i])
  dataf[l,]$x5=d$DPII[1]
  dataf[l,]$x6=d$DEBI[1]
}

Ni=rep(0,length(unique(dataf$cluster)))
clusteru=unique(dataf$cluster)


Nij=matrix(0,length(unique(dataf$cluster)),J)
for(i in 1:length(unique(dataf$cluster))){
  l=dataf[dataf$cluster==clusteru[i],]
  Ni[i]=length(unique(l$id))
  for(j in 1:J){
    l=which((dataf$cluster==clusteru[i])&(dataf$time==j))
    Nij[i,j]=length(l)
  }
}
I=length(clusteru)
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
m=2
samples=samplesplit(m,Z)
dataf$sp=1
for(i in 1:I){
  dataf[dataf$cluster==clusteru[i],]$sp=samples[i]
}
dataf$g=1
for(i in 1:m){
  for(j in 1:J){
    train=which(dataf$sp!=i&dataf$time==j)
    Ytrain=dataf$y[train]
    Xtrain=dataf[train,7:11]
    test=which(dataf$sp==i&dataf$time==j)
    Ytest= dataf$y[test]
    Xtest=dataf[test,7:11]
    
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
D=do.call(rbind,Dmu)
dataf$residual=dataf$y-dataf$g
l=which(S==1)
D=D[l,]
for(i in 1:I){
  for(j in 1:J){
    l=which(dataf$cluster==clusteru[i]&dataf$time==j)
    if(length(l)!=0){
      D[l,]=D[l,]/sqrt(Nij[i,j])
      dataf$residual[l]=dataf$residual[l]/sqrt(Nij[i,j])
    }
  }
}


Dmu=D
re=dataf$residual



Dmu_list <- split(seq_len(nrow(Dmu)), dataf$cluster)   
Dmu <- lapply(Dmu_list, function(idx) Dmu[idx, , drop = FALSE])
resi_listf=split(re,dataf$cluster)
Dsquare=Reduce("+",Map(function(a, b) t(a) %*% b, Dmu, Dmu)) 
Dsinv=solve(Dsquare)
Dvsresidual=Reduce("+",Map(function(a, b) t(a) %*% b, Dmu, resi_listf)) 
betaest=Dsinv%*%Dvsresidual
phi=Map(function(a, b) {t(a) %*% (b-a%*%betaest)}, Dmu, resi_listf)
meat=Reduce("+",Map(function(a, b) a %*% t(b), phi, phi)) 
vmatrix=Dsinv%*%meat%*%Dsinv

betaest
betaest1=betaest
vmatrix

idlist=which(S==1)
idlist=split(idlist,dataf$cluster)
for(i in 1:I){
  idlist[[i]]=idlist[[i]]-idlist[[i]][1]+1
  idlist[[i]]=idlist[[i]]%%(Ni[i]*J)
  idlist[[i]][idlist[[i]]==0]=Ni[i]*J
}
Q1=list()
Q2=list()
Q3=list()
Q4=list()
for(i in 1:I){
  num=length(idlist[[i]])
  l=matrix(0,num,num)
  diag(l)=1
  Q1[[i]]=l
  l2 <- matrix(1,num,num)
  Q2[[i]]=l2
  
  block_id=(idlist[[i]] - 1) %/% Ni[i]
  l3= outer(block_id, block_id, "==") * 1
  Q3[[i]]=l3
  mod_vals=(idlist[[i]]) %% Ni[i]
  Q4[[i]]=outer(mod_vals, mod_vals, "==")*1
}
Qfunction=function(i,t=1){
  q=list(Q3[[i]],Q2[[i]],Q1[[i]])
  return(q)
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
cplus1=map(1:I, function(i) {
  phii=phibeta[[i]][,1]%*%t(phiiold[[i]])+phiiold[[i]]%*%t(phibeta[[i]][,1])
  return (phii)
})
cplus2=map(1:I, function(i) {
  phii=phibeta[[i]][,2]%*%t(phiiold[[i]])+phiiold[[i]]%*%t(phibeta[[i]][,2])
  return (phii)
})
phibeta=Reduce("+",phibeta)
cplus1=Reduce("+",cplus1)
cplus2=Reduce("+",cplus2)


phiiold=Reduce("+",phiiold)
phivarinverse=t(phibeta)%*%CIinverse%*%phibeta
phivar=solve(phivarinverse)
G1=phivar%*%t(phibeta)%*%CIinverse%*%cplus1%*%CIinverse%*%phiiold
G2=phivar%*%t(phibeta)%*%CIinverse%*%cplus2%*%CIinverse%*%phiiold


G=cbind(G1,G2)
fv1=(diag(1,nrow(G))+G)%*%phivar
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
    t(Dmu[[i]])%*%Qq%*%solve(-Oi[[i]]+diag(1,nrow(Oi[[i]])))%*%(resi_listf[[i]]-Dmu[[i]]%*%betaest)
  })
  phii=do.call(rbind,phii)
  return (phii)
})
CInew=Reduce("+",Map(function(a, b) a %*% t(b), phiinew, phiinew))
fv=(diag(1,nrow(G))+G)%*%phivar%*%t(phibeta)%*%CIinverse%*%CInew%*%CIinverse%*%
  phibeta%*%phivar%*%t(diag(1,nrow(G))+G)

b1=mean(betaest1)
b2=mean(betaest)
sss=c(1/J,1/J)
v1=t(sss)%*%vmatrix%*%sss
v2=t(sss)%*%phivar%*%sss




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


data=read.csv()
datab=read.csv()
J=2
data=data[order(data$TInStudy),]
data=data[order(data$person_id),]
table(data$person_id)
t=table(data$person_id)
l=which(t>1)

ch=names(l)
data=data[data$person_id%in%ch,]

data$time=rep(1,length(data$cluster_group_id))
data[data$TInStudy<=50,]$time=0
data[(data$TInStudy>50)&(data$TInStudy<=140),]$time=1
data[(data$TInStudy>140)&(data$TInStudy<=230),]$time=2
data[(data$TInStudy>230)&(data$TInStudy<=320),]$time=3
data[(data$TInStudy>320)&(data$TInStudy<=400),]$time=4
data[data$TInStudy>400,]$time=5


s=duplicated(data$person_id)
l=which(s==FALSE)
l
database=data[l,]
l=which(data$time>0)
datal=data[l,]

length(unique(database$person_id))
length(unique(datal$person_id))

database=database[database$person_id%in%unique(datal$person_id),]
length(unique(database$person_id))



X=data.frame(id=database$person_id,x1=database$death,x2=database$sex,x3=database$age,x4=database$Total)
Y=data.frame(cp=datal$cluster_group_id,cluster=datal$cluster_id,id=datal$person_id,time=datal$time,y=datal$Total)

datau=Y%>%left_join(X,by="id")
datau=datau[order(datau$time),]
datau=datau[order(datau$cluster),]

database=database[order(database$time),]
database=database[order(database$cluster_id),]


dataf=NULL
id=unique(database$person_id)
S=rep(1,length(id)*J)
t=1

for(i in 1:length(id)){
  for(j in 1:J){
    l=which((datau$id==id[i])&(datau$time==j))
    if(length(l)==1){
      dataf=rbind(dataf,datau[l,])
    }
    if(length(l)>1){
      d=datau[l[1],]
      d$y=mean(datau$y[l])
      dataf=rbind(dataf,d)
    }
  }
}

dataf=dataf[order(dataf$id),]
dataf=dataf[order(dataf$time),]
dataf=dataf[order(dataf$cluster),]
dataf=dataf[dataf$time%in%c(1,2),]
J=max(dataf$time)

S=rep(1,length(id)*J)
t=1
for(ss in 1:length(unique(dataf$cluster))){
  idc=unique(dataf[dataf$cluster==unique(dataf$cluster)[ss],]$id)
  for(j in 1:J){
    for(i in 1:length(idc)){
      l=which((dataf$id==idc[i])&(dataf$time==j))
      if(length(l)==0){
        S[t]=0
      }
      t=t+1
    }
  }
}
l=(!duplicated(dataf$cluster))
Z=dataf$cp[l]

datab=datab[datab$person_id%in%dataf$id,]
iid=unique(datab$person_id)
dataf$x5=1
dataf$x6=1
for(i in 1:length(iid)){
  l=which(datab$person_id==iid[i])
  d=datab[l,]
  d=d[order(d$TInStudy),]
  l=which(dataf$id==iid[i])
  dataf[l,]$x5=d$DPII[1]
  dataf[l,]$x6=d$DEBI[1]
}

Ni=rep(0,length(unique(dataf$cluster)))
clusteru=unique(dataf$cluster)


Nij=matrix(0,length(unique(dataf$cluster)),J)
for(i in 1:length(unique(dataf$cluster))){
  l=dataf[dataf$cluster==clusteru[i],]
  Ni[i]=length(unique(l$id))
  for(j in 1:J){
    l=which((dataf$cluster==clusteru[i])&(dataf$time==j))
    Nij[i,j]=length(l)
  }
}
I=length(clusteru)
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
m=2
samples=samplesplit(m,Z)
dataf$sp=1
for(i in 1:I){
  dataf[dataf$cluster==clusteru[i],]$sp=samples[i]
}
dataf$g=1
for(i in 1:m){
  for(j in 1:J){
    train=which(dataf$sp!=i&dataf$time==j)
    Ytrain=dataf$y[train]
    Xtrain=dataf[train,7:11]
    test=which(dataf$sp==i&dataf$time==j)
    Ytest= dataf$y[test]
    Xtest=dataf[test,7:11]
    
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
D=do.call(rbind,Dmu)
dataf$residual=dataf$y-dataf$g
l=which(S==1)
D=D[l,]
for(i in 1:I){
  for(j in 1:J){
    l=which(dataf$cluster==clusteru[i]&dataf$time==j)
    if(length(l)!=0){
      D[l,]=D[l,]/sqrt(Nij[i,j])
      dataf$residual[l]=dataf$residual[l]/sqrt(Nij[i,j])
    }
  }
}


Dmu=D
re=dataf$residual



Dmu_list <- split(seq_len(nrow(Dmu)), dataf$cluster)   
Dmu <- lapply(Dmu_list, function(idx) Dmu[idx, , drop = FALSE])
resi_listf=split(re,dataf$cluster)
Dsquare=Reduce("+",Map(function(a, b) t(a) %*% b, Dmu, Dmu)) 
Dsinv=solve(Dsquare)
Dvsresidual=Reduce("+",Map(function(a, b) t(a) %*% b, Dmu, resi_listf)) 
betaest=Dsinv%*%Dvsresidual
phi=Map(function(a, b) {t(a) %*% (b-a%*%betaest)}, Dmu, resi_listf)
meat=Reduce("+",Map(function(a, b) a %*% t(b), phi, phi)) 
vmatrix=Dsinv%*%meat%*%Dsinv

betaest
betaest1=betaest
vmatrix

idlist=which(S==1)
idlist=split(idlist,dataf$cluster)
for(i in 1:I){
  idlist[[i]]=idlist[[i]]-idlist[[i]][1]+1
  idlist[[i]]=idlist[[i]]%%(Ni[i]*J)
  idlist[[i]][idlist[[i]]==0]=Ni[i]*J
}
Q1=list()
Q2=list()
Q3=list()
Q4=list()
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
Qfunction=function(i,t=1){
  q=list(Q1[[i]],Q2[[i]],Q4[[i]])
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
betaest
phivar

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
cplus1=map(1:I, function(i) {
  phii=phibeta[[i]][,1]%*%t(phiiold[[i]])+phiiold[[i]]%*%t(phibeta[[i]][,1])
  return (phii)
})
cplus2=map(1:I, function(i) {
  phii=phibeta[[i]][,2]%*%t(phiiold[[i]])+phiiold[[i]]%*%t(phibeta[[i]][,2])
  return (phii)
})
cplus3=map(1:I, function(i) {
  phii=phibeta[[i]][,3]%*%t(phiiold[[i]])+phiiold[[i]]%*%t(phibeta[[i]][,3])
  return (phii)
})
phibeta=Reduce("+",phibeta)
cplus1=Reduce("+",cplus1)
cplus2=Reduce("+",cplus2)
cplus3=Reduce("+",cplus3)

phiiold=Reduce("+",phiiold)
phivarinverse=t(phibeta)%*%CIinverse%*%phibeta
phivar=solve(phivarinverse)
G1=phivar%*%t(phibeta)%*%CIinverse%*%cplus1%*%CIinverse%*%phiiold
G2=phivar%*%t(phibeta)%*%CIinverse%*%cplus2%*%CIinverse%*%phiiold
G3=phivar%*%t(phibeta)%*%CIinverse%*%cplus3%*%CIinverse%*%phiiold

G=cbind(G1,G2,G3)
fv1=(diag(1,nrow(G))+G)%*%phivar%*%(diag(1,nrow(G))+G)
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
    t(Dmu[[i]])%*%Qq%*%solve(-Oi[[i]]+diag(1,nrow(Oi[[i]])))%*%(resi_listf[[i]]-Dmu[[i]]%*%betaest)
  })
  phii=do.call(rbind,phii)
  return (phii)
})
CInew=Reduce("+",Map(function(a, b) a %*% t(b), phiinew, phiinew))
fv=(diag(1,nrow(G))+G)%*%phivar%*%t(phibeta)%*%CIinverse%*%CInew%*%CIinverse%*%
  phibeta%*%phivar%*%t(diag(1,nrow(G))+G)

b1=mean(betaest1)
b2=mean(betaest)
sss=c(1/3,1/3,1/3)
v1=t(sss)%*%vmatrix%*%sss
v2=t(sss)%*%fv%*%sss



