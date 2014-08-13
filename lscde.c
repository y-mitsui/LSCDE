#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "matrix.h"
#include "mesch12b/matrix.h"
#include "mesch12b/matrix2.h"
#include "lscde.h"

#define Malloc(type,n) (type *)malloc((n)*sizeof(type))
#define Calloc(type,n) (type *)calloc(1,(n)*sizeof(type))

void printMatrix(MAT *a,char *name);
MAT *m_pixel(MAT *a,MAT *b,MAT *out,char command);
MAT *m_get_zero(int m,int n);
MAT *m_get_ident(int m,int n);
MAT * m_max(MAT *source,int n,MAT *target);
MAT *repmat(MAT *source,int repeat_m,int repeat_n,MAT *out);
MAT *m_filter(MAT *a,MAT *out,double (*filter)(double));
MAT *doble2MAT(double *source,int num,int isRowVector);
MAT* i_index(int *arr,int num,int (*condition)(int ,void *,int),void *arg);
MAT *m_index(MAT *source,int (*m_condition)(void *,int),int (*n_condition)(void *,int,int),void *arg);

void randperm(int *result,int n){
	int i;
	for(i=0;i<n;i++){
		result[i]=rand()%n;
	}
}
double kernel(double x,double Xcenter,double y,double yCenter,double sigma){
	double param=2*sigma*sigma;
	return exp(-sqrt((x-Xcenter)*(x-Xcenter))/param)*exp(-sqrt((y-yCenter)*(y-yCenter))/param);
}
MAT* mylinsolve(MAT *a,MAT *b){
	return m_mlt(m_inverse(a,NULL),b,NULL);
}
MAT *getDistance(MAT *train,MAT *center,int numSample,int b){
	MAT *repmatA=repmat(m_pixel(train,train,NULL,'*'),b,1,NULL);
	MAT *temp=m_transp(m_pixel(center,center,NULL,'*'),NULL);
	MAT *repmatB=repmat(temp,1,numSample,NULL);
	MAT *uTrans2x=sm_mlt(2.0,m_transp(center,NULL),NULL);
	MAT *targetC=m_mlt(uTrans2x,train,NULL);
	return m_sub(m_add(repmatA,repmatB,NULL),targetC,NULL);
}
double *
linspace (float x1, float x2, int n)
{
	int i;
  if (n < 1) n = 1;

  double *retval=malloc(n*sizeof(double));

  float delta = (x2 - x1) / (n - 1);
  retval[0] = x1;
  for (i = 1; i < n-1; i++)
    retval[i] = x1 + i*delta;
  retval[n-1] = x2;

  return retval;
}
double *logspace(double from,double to,int num){
	int i;
	double *r=malloc(num*sizeof(double));

	//double par=diff/num;


	double *lin=linspace(from,to,num);
	for(i=0;i<num;i++){
		r[i]=pow(10,lin[i]);
	}


	return r;
}
MAT *makeLearning(){
	return NULL;
}
MAT *make_phi(MAT *dist2,double sigma,double n){
	MAT *temp=sm_mlt(1.0/(n*sigma*sigma),dist2,NULL);
	MAT *minus=m_pixel(m_get_zero(temp->m,temp->n),temp,NULL,'-');
	return m_filter(minus,NULL,exp);
}
int basicFilter(int data,void *arg,int n){
	void **arg2=(void **)arg;
	int *x=arg2[0];
	int *y=arg2[1];
	return x[n]==*y;
}
int basicFilterNot(int data,void *arg,int n){
	void **arg2=(void **)arg;
	int *x=arg2[0];
	int *y=arg2[1];
	return x[n]!=*y;
}
int basicMatFilter(void *arg,int m,int n){
	int i;
	MAT *ma=arg;
	for(i=0;i<ma->n;i++) if((int)ma->me[0][i]-1==n) return 1;
	return 0;
}
#define NUM_SIGMA 9
lscde* lscdeModel(double *xTrain,double *yTrain,int xDim,int yDim,int numSample){
	
	int b=(numSample < 100) ? numSample : 100;
	int i,j,k,l,m;

	int *randIndex=Malloc(int,numSample);
	int *cv_index=Malloc(int,numSample);
	int *cv_split=Malloc(int,numSample); //floor([0:n-1]*fold./n)+1;
	int fold=5;

	for(i=0;i<numSample;i++){
		cv_split[i]=(int)((double)i*(double)fold/(double)numSample);
	}

	double *sigma_list=logspace(-1.5,1.5,NUM_SIGMA);
	double sigma;
	double *lambda_list=logspace(-3.0,1.0,NUM_SIGMA);
	double lambda;
	double cc=2.2204e-16;


	MAT *xCenters=m_get(1,b);
	MAT *yCenters=m_get(1,b);
	
	MAT *alphat;
	MAT **Phibar_cv=malloc(fold*sizeof(MAT *));

	randperm(randIndex,numSample);
	randperm(cv_index,numSample);

	/*fload("../LSCDE_matlab/rand_index.csv",randIndex);
	fload("../LSCDE_matlab/cv_index.csv",cv_index);*/
	
	
	for(i=0;i<b;i++){
		xCenters->me[0][i]=xTrain[randIndex[i]-1];
		yCenters->me[0][i]=yTrain[randIndex[i]-1];
	}
	MAT *x_train=doble2MAT(xTrain,numSample,0);
	MAT *y_train=doble2MAT(yTrain,numSample,0);
	
	
	MAT *xu_dist2=getDistance(x_train,xCenters,numSample,b);
	MAT *yv_dist2=getDistance(y_train,yCenters,numSample,b);

	MAT *repmatA=repmat(m_pixel(yCenters,yCenters,NULL,'*'),b,1,NULL);
	MAT *repmatB=repmat(m_transp(m_pixel(yCenters,yCenters,NULL,'*'),NULL),1,b,NULL);
	MAT *uTrans2x=sm_mlt(2.0,m_transp(yCenters,NULL),NULL);
	MAT *targetC=m_mlt(uTrans2x,yCenters,NULL);
	MAT *vv_dist2=m_sub(m_add(repmatA,repmatB,NULL),targetC,NULL);
	MAT *score_cv=m_get(NUM_SIGMA,NUM_SIGMA);
	for(i=0;i<NUM_SIGMA;i++){
		sigma=sigma_list[i];
		MAT *phi_xu=make_phi(xu_dist2,sigma,2.0);
		MAT *phi_yv=make_phi(yv_dist2,sigma,2.0);
		MAT *phi_zw=m_pixel(phi_xu,phi_yv,NULL,'*');
		MAT *phi_vv=make_phi(vv_dist2,sigma,4.0);
		MAT **Phibar_cv=malloc(fold*sizeof(MAT*));
		for(j=0;j<fold;j++){
			void *arg[2]={cv_split,&j};
			MAT *cv_index_part=i_index(cv_index,numSample,basicFilter,arg);
			MAT *tmp=mn_part(phi_xu,cv_index_part);
			MAT *tmp22=m_mlt(tmp,m_transp(tmp,NULL),NULL);
			Phibar_cv[j]=m_pixel(sm_mlt(pow(sqrt(M_PI)*sigma,yDim),phi_vv,NULL),tmp22,NULL,'*');
		}
		for(j=0;j<NUM_SIGMA;j++){
			lambda=lambda_list[j];
			double score_tmp=0.0;
			for(k=0;k<fold;k++){
				MAT *sum=m_get_zero(Phibar_cv[0]->m,Phibar_cv[0]->n);
				for(l=0;l<fold;l++){
					if(l!=k){
						m_add(sum,Phibar_cv[l],sum);
					}
				}
				int total=0;
				for(l=0;l<numSample;l++){
					if(cv_split[l]!=k){
						total++;
					}
				}
				MAT *matA=sm_mlt(1.0/(double)total,sum,NULL);
				
				MAT *matB=m_get_ident(b,b);
				MAT *matC=sm_mlt(lambda,matB,NULL);
				MAT *matD=m_add(matA,matC,NULL);

				void *arg[2]={cv_split,&k};
				MAT *cv_index_part=i_index(cv_index,numSample,basicFilterNot,arg);
				MAT *matE=mn_part(phi_zw,cv_index_part);
				
				MAT *matF=m_get(b,1);
				for(m=0;m<b;m++){
					float mean=0.0;
					for(l=0;l<matE->n;l++){
						mean+=matE->me[m][l];
					}
					mean/=matE->n;
					matF->me[m][0]=mean;
				}
				
				MAT *alphah_cv=mylinsolve(matD,matF);
				
				m_max(alphah_cv,0,alphah_cv);

				arg[1]=&k;
				cv_index_part=i_index(cv_index,numSample,basicFilter,arg);
				MAT *tmp=mn_part(phi_xu,cv_index_part);
				MAT *normalization_cv=m_mlt(sm_mlt(pow(sqrt(2*M_PI)*sigma,yDim),m_transp(alphah_cv,NULL),NULL),tmp,NULL);
				m_max(normalization_cv,cc,normalization_cv);

				tmp=mn_part(phi_zw,cv_index_part);
				MAT *ph_cv=m_pixel(m_mlt(m_transp(alphah_cv,NULL),tmp,NULL),normalization_cv,NULL,'/');
				
				for(l=0;l<ph_cv->m;l++){
					for(m=0;m<ph_cv->n;m++){
						ph_cv->me[l][m]+=cc;
					}
				}
				double mean=0.0;
				for(l=0;l<ph_cv->m;l++){
					for(m=0;m<ph_cv->n;m++){
						mean+=log(ph_cv->me[l][m]);
					}
				}
				mean/=ph_cv->n;
				score_tmp+=-mean;

			}
			score_cv->me[i][j]=score_tmp/fold;
		}
	}
	
	int *idx=Malloc(int,score_cv->m);
	double *min=Malloc(double,score_cv->m);
	for(i=0;i<score_cv->m;i++){
		min[i]=999999.0;
		for(j=0;j<score_cv->n;j++){
			if(min[i] > score_cv->me[i][j]){
				idx[i]=j;
				min[i]=score_cv->me[i][j];
			}
		}
	}
	double sigma_min=9999.0;
	int sigma_index=0;
	for(i=0;i<score_cv->m;i++){
		if(sigma_min > min[i]){
			sigma_index=i;
			sigma_min=min[i];
		}
	}
	sigma=sigma_list[sigma_index];
	lambda=lambda_list[idx[sigma_index]];
	printf("%lf %lf\n",sigma,lambda);
	MAT *phi_xu=make_phi(xu_dist2,sigma,2.0);
	MAT *phi_yv=make_phi(yv_dist2,sigma,2.0);
	MAT *phi_zw=m_pixel(phi_xu,phi_yv,NULL,'*');
	MAT *phi_vv=make_phi(vv_dist2,sigma,4.0);
	MAT *tmp22=m_mlt(phi_xu,m_transp(phi_xu,NULL),NULL);
	MAT *Phibar=m_pixel(sm_mlt(pow(sqrt(M_PI)*sigma,yDim),phi_vv,NULL),tmp22,NULL,'*');
	MAT *tmp=sm_mlt(lambda,m_get_ident(b,b),NULL);
	MAT *tmpB=m_add(sm_mlt(1.0/(double)numSample,Phibar,NULL),tmp,NULL);

	MAT *matF=m_get(phi_zw->m,1);
	for(m=0;m<phi_zw->m;m++){
		float mean=0.0;
		for(l=0;l<phi_zw->n;l++){
			mean+=phi_zw->me[m][l];
		}
		mean/=phi_zw->n;
		matF->me[m][0]=mean;
	}
	alphat=mylinsolve(tmpB,matF);
	m_max(alphat,0,alphat);
	//printMatrix(alphat,"alphat");

	lscde *model=Malloc(lscde,1);
	model->coefficient=alphat;
	model->xCenters=xCenters;
	model->yCenters=yCenters;
	model->sigma=sigma;

	//exit(1);
	return model;

	/*MAT *temp=m_transp(m_pixel(center,center,NULL,'*'),NULL);
	MAT *repmatB=repmat(temp,1,numSample,NULL);
	MAT *uTrans2x=sm_mlt(2.0,m_transp(center,NULL),NULL);
	MAT *targetC=m_mlt(uTrans2x,train,NULL);
	return m_sub(m_add(repmatA,repmatB,NULL),targetC,NULL);*/

	
	/*for(i=0;i<b;i++){
		for(j=0;j<numSample;j++){
			repmatXX->me[i][j]=tmp1NX->me[0][j];
			repmatYY->me[i][j]=tmp1NY->me[0][j];
			repmatUU->me[i][j]=tmp1bU->me[0][i];
			repmatVV->me[i][j]=tmp1bV->me[0][i];
		}
	}
	MAT *xu_dist2=getDistance(x_train,xCenters,repmatXX,repmatUU);
	printMatrix(xu_dist2,"xu_dist2");
	MAT *yv_dist2=getDistance(y_train,yCenters,repmatYY,repmatVV);
	for(i=0;i<b;i++){
		for(j=0;j<b;j++){
			repmatV2->me[i][j]=tmp1bV->me[0][j];
			repmatV2T->me[j][i]=tmp1bV->me[0][j];
		}
	}
	MAT *vv_dist2=getDistance(yCenters,yCenters,repmatV2,repmatV2T);
	MAT *score_cv=m_get(9,9);
	m_zero(score_cv);

	for(i=0;i<9;i++){
		sigma=sigma_list[i];

		MAT *phi_xu=m_get(b,numSample);
		MAT *tmpB=sm_mlt(1.0/(2.0*sigma*sigma),xu_dist2,NULL);
		for(i=0;i<b;i++){
			for(j=0;j<numSample;j++){
				phi_xu->me[i][j]=exp(-tmpB->me[i][j]);
			}
		}

		MAT *phi_yv=m_get(b,numSample);
		MAT *tmpC=sm_mlt(1.0/(2.0*sigma*sigma),yv_dist2,NULL);
		for(i=0;i<b;i++){
			for(j=0;j<numSample;j++){
				phi_yv->me[i][j]=exp(-tmpC->me[i][j]);
			}
		}

		MAT *phi_zw=m_get(b,numSample);
		for(i=0;i<b;i++){
			for(j=0;j<numSample;j++){
				phi_zw->me[i][j]=phi_xu->me[i][j]*phi_yv->me[i][j];
			}
		}

		MAT *phi_vv=m_get(b,b);
		MAT *tmpD=sm_mlt(1.0/(4.0*sigma*sigma),vv_dist2,NULL);
		for(i=0;i<b;i++){
			for(j=0;j<b;j++){
				phi_vv->me[i][j]=exp(-tmpD->me[i][j]);
			}
		}
		for(j=0;j<fold;j++){
			MAT *tmp=m_get(phi_xu->m,numSample/fold+1);
			for(k=0;k<phi_xu->m;k++){
				int tmpCt=0;
				for(l=0;l<phi_xu->n;l++){
					if(cv_split[l]==j){
						tmp->me[k][tmpCt++]=phi_xu->me[k][l];
					}
				}
			}

			MAT *tmp2=m_mlt(tmp,m_transp(tmp,NULL),NULL);

			Phibar_cv[j]=m_get(b,b);
			for(k=0;k<b;k++){
				for(l=0;l<b;l++){
					Phibar_cv[j]->me[k][l]=pow(sqrt(M_PI)*sigma,yDim)*phi_vv->me[k][l]*tmp2->me[k][l];
				}
			}
			//printMatrix(Phibar_cv[j],"Phibar_cv[j]");

		}
		exit(1);
		for(j=0;j<9;j++){
			lambda=lambda_list[j];
			double score_tmp=0.0;
			for(k=0;k<fold;k++){
				MAT *sum=m_get_zero(b,b);
				for(l=0;l<fold;l++){
					if(l!=k){
						m_add(sum,Phibar_cv[l],sum);
					}
				}
				printMatrix(sum,"sum");
				int total=0;
				for(l=0;l<numSample;l++){
					if(cv_split[l]!=k){
						total++;
					}
				}
				MAT *matA=sm_mlt(1.0/(double)total,sum,NULL);
				MAT *matB=m_get_ident(b,b);
				MAT *matC=sm_mlt(lambda,matB,NULL);
				printf("lambda:%lf\n",lambda);
				MAT *matD=m_add(matA,matC,NULL);
				MAT *matE=m_get(b,total);
				int me=0;
				for(l=0;l<numSample;l++){
					if(cv_split[l]!=k){
						for(m=0;m<b;m++){
							matE->me[m][me]=phi_zw->me[m][cv_index[l]];
						}
						me++;
					}
				}
				MAT *matF=m_get(b,1);
				for(m=0;m<b;m++){
					double mean=0.0;
					for(l=0;l<total;l++){
						mean+=matE->me[m][l];
					}
					mean/=numSample;
					matF->me[m][0]=mean;
				}
				MAT *alphat=mylinsolve(matD,matF);
				//printMatrix(alphat,"alphat");
				//puts("------------------");
				m_max(alphat,0,alphat);
				
				MAT *tmp=m_get(phi_xu->m,numSample/fold+1);

				for(l=0;l<phi_xu->m;l++){
					int tmpCt=0;
					for(m=0;m<phi_xu->n;m++){
						if(cv_split[m]==j){
							tmp->me[l][tmpCt++]=phi_xu->me[l][m];
						}
					}
				}
				MAT *normalization_cv=m_mlt(sm_mlt(pow(sqrt(2*M_PI)*sigma,yDim),m_transp(alphat,NULL),NULL),tmp,NULL);
				MAT *matG=m_get(b,numSample-total);
				me=0;
				for(l=0;l<numSample;l++){
					if(cv_split[l]==k){
						for(m=0;m<b;m++){
							matG->me[m][me]=phi_zw->me[m][cv_index[l]];
						}
						me++;
					}
				}
				MAT *ph_cv=m_pixel(m_mlt(m_transp(alphat,NULL),matG,NULL),normalization_cv,NULL,'/');
				for(l=0;l<ph_cv->m;l++){
					for(m=0;m<ph_cv->n;m++){
						ph_cv->me[l][m]+=cc;
					}
				}
				double mean=0.0;
				for(l=0;l<ph_cv->m;l++){
					for(m=0;m<ph_cv->n;m++){
						mean+=log(ph_cv->me[l][m]);
					}
				}
				mean/=ph_cv->n;
				score_tmp+=-mean;

			}
			score_cv->me[i][j]=score_tmp/fold;
		}


		/*MAT *Phibar=sm_mlt(pow(sqrt(M_PI)*sigma,1),phi_vv,NULL);
		MAT *tmpE=m_mlt(phi_xu,m_transp(phi_xu,NULL),NULL);
		for(i=0;i<b;i++){
			for(j=0;j<b;j++){
				Phibar->me[i][j]*=tmpE->me[i][j];
			}
		}

		MAT *PhibarDived=sm_mlt(1.0/(double)numSample,Phibar,NULL);
		double lambda=1.0000e-003;
		MAT *ident=m_get(b,b);
		MAT *tmpF=sm_mlt(lambda,m_ident(ident),NULL);
		MAT *tmpG=m_add(tmpF,PhibarDived,NULL);*/


		/*for(j=0;j<fold;j++){
		 *
			int t=cv_index[cv_split[j]];
			for(k=0;k<phi_xu->m;k++){
				tmp->me[0][k]=phi_xu->me[k][t];
			}
			for(k=0;k<phi_xu->m;k++){
				Phibar_cv[j]->me[0][k]=pow(sqrt(M_PI)*sigma,yDim)*phi_vv->me[0][k]*tmp2->me[0][k];
			}

		}

		      tmp=phi_xu(:,cv_index(cv_split==k));
		      Phibar_cv(:,:,k)=(sqrt(pi)*sigma)^d_y*phi_vv.*(tmp*tmp');
		end % for fold*/

		/*MAT *phi_zwMean=m_get(b,1);
		for(i=0;i<b;i++){
			double sTmp=0.0;
			for(j=0;j<numSample;j++){
				sTmp+=phi_xu->me[i][j]*phi_yv->me[i][j];
			}
			phi_zwMean->me[i][0]=sTmp/(double)numSample;
		}
		alphat=mylinsolve(tmpG,phi_zwMean);
		for(i=0;i<b;i++){
			alphat->me[i][0]=(alphat->me[i][0] < 0 ) ? 0: alphat->me[i][0];
		}*\/

		m_free(phi_xu);
		m_free(tmpB);
		m_free(phi_yv);
		m_free(tmpC);
		m_free(phi_vv);
		m_free(tmpD);
		/*m_free(Phibar);
		m_free(tmpE);
		m_free(PhibarDived);
		m_free(tmpF);
		m_free(tmpG);
		m_free(phi_zwMean);*\/
	}
	exit(1);
	int *idx=Malloc(int,score_cv->m);
	double *min=Malloc(double,score_cv->m);
	for(i=0;i<score_cv->m;i++){
		min[i]=999999.0;
		for(j=0;j<score_cv->n;j++){
			if(min[i] > score_cv->me[i][j]){
				idx[i]=j;
				min[i]=score_cv->me[i][j];
			}
		}
	}
	double sigma_min=9999.0;
	int sigma_index=0;
	for(i=0;i<score_cv->m;i++){
		if(sigma_min > min[i]){
			sigma_index=i;
			sigma_min=min[i];
		}
	}
	double fsigma=sigma_list[sigma_index];
	double flambda=lambda_list[idx[sigma_index]];

			MAT *phi_xu=m_get(b,numSample);
			MAT *tmpB=sm_mlt(1.0/(2.0*fsigma*fsigma),xu_dist2,NULL);
			for(i=0;i<b;i++){
				for(j=0;j<numSample;j++){
					phi_xu->me[i][j]=exp(-tmpB->me[i][j]);
				}
			}

			MAT *phi_yv=m_get(b,numSample);
			MAT *tmpC=sm_mlt(1.0/(2.0*fsigma*fsigma),yv_dist2,NULL);
			for(i=0;i<b;i++){
				for(j=0;j<numSample;j++){
					phi_yv->me[i][j]=exp(-tmpC->me[i][j]);
				}
			}

			MAT *phi_zw=m_get(b,numSample);
			for(i=0;i<b;i++){
				for(j=0;j<numSample;j++){
					phi_zw->me[i][j]=phi_xu->me[i][j]*phi_yv->me[i][j];
				}
			}

			MAT *phi_vv=m_get(b,b);
			MAT *tmpD=sm_mlt(1.0/(4.0*fsigma*fsigma),vv_dist2,NULL);
			for(i=0;i<b;i++){
				for(j=0;j<b;j++){
					phi_vv->me[i][j]=exp(-tmpD->me[i][j]);
				}
			}
					MAT *sum=m_get(b,b);
					m_zero(sum);
					for(l=0;l<fold;l++){
						if(l!=k){
							m_add(sum,Phibar_cv[l],sum);
						}
					}
					int total=0;
					for(l=0;l<numSample;l++){
						if(cv_split[l]!=k){
							total++;
						}
					}
					MAT *matA=sm_mlt(1.0/(double)total,sum,NULL);
					MAT *matB=m_get(b,b);
					m_ident(matB);
					MAT *matC=sm_mlt(flambda,matB,NULL);
					MAT *matD=m_add(matA,matC,NULL);
					MAT *matE=m_get(b,total);
					int me=0;
					for(l=0;l<numSample;l++){
						if(cv_split[l]!=k){
							for(m=0;m<b;m++){
								matE->me[m][me]=phi_zw->me[m][cv_index[l]];
							}
							me++;
						}
					}
					MAT *matF=m_get(b,1);
					for(m=0;m<b;m++){
						double mean=0.0;
						for(l=0;l<total;l++){
							mean+=matE->me[m][l];
						}
						mean/=numSample;
						matF->me[m][0]=mean;
					}
					alphat=mylinsolve(matD,matF);

	printMatrix(alphat,"alphat");
	lscde *model=Malloc(lscde,1);
	model->coefficient=alphat;
	model->xCenters=xCenters;
	model->yCenters=yCenters;
	model->sigma=sigma;

	m_free(tmp1NX);
	m_free(tmp1NY);
	m_free(tmp1bU);
	m_free(tmp1bV);
	m_free(repmatXX);
	m_free(repmatYY);
	m_free(repmatUU);
	m_free(repmatVV);
	m_free(repmatV2);
	m_free(repmatV2T);
	m_free(x_train);
	m_free(y_train);
	m_free(xu_dist2);
	m_free(yv_dist2);
	m_free(vv_dist2);
	

	free(randIndex);	

	return model;*/
}

/*
	compute P(Y|X)
*/
MAT *kernel_Gaussian(MAT *x,MAT *center,double sigma){
	int i,j;
	MAT *center2=m_pixel(center,center,NULL,'*');
	MAT *x2=m_transp(m_pixel(x,x,NULL,'*'),NULL);
	
	MAT *tmpA=m_get(x->n,center->n);
	MAT *tmpB=m_get(x2->m,center->n);
	for(i=0;i<x->n;i++){
		for(j=0;j<center->n;j++){
			tmpA->me[i][j]=center2->me[0][j];
		}
	}
	
	
	for(j=0;j<center->n;j++){
		for(i=0;i<x2->m;i++){
			tmpB->me[i][j]=x2->me[i][0];
		}
	}

	MAT *tmp=m_mlt(sm_mlt(2.0,m_transp(x,NULL),NULL),center,NULL);
	MAT *distance2=m_sub(m_add(tmpA,tmpB,NULL),tmp,NULL);
	MAT *r=m_get(distance2->m,distance2->n);

	for(i=0;i<distance2->m;i++){
		for(j=0;j<distance2->n;j++){
			r->me[i][j]=exp(-distance2->me[i][j]/(2*sigma*sigma));
		}
	}

	free(center2);
	free(tmpA);
	free(tmpB);
	free(tmp);
	free(distance2);
	return r;
}
MAT* lscdeConditionalDensity(lscde *model,double *X,double *Y,int numTest){
	MAT *x,*y;
	int i,j,dy=1;

	x=m_get(1,numTest);
	y=m_get(1,numTest);
	for(i=0;i<numTest;i++){
		x->me[0][i]=X[i];
		y->me[0][i]=Y[i];
	}

	MAT *phi_xu_test=m_transp(kernel_Gaussian(x,model->xCenters,model->sigma),NULL);
	MAT *phi_yv_test=m_transp(kernel_Gaussian(y,model->yCenters,model->sigma),NULL);
	MAT *phi_zw_test=m_get(phi_xu_test->m,phi_xu_test->n);
	for(i=0;i<phi_xu_test->m;i++){
		for(j=0;j<phi_xu_test->n;j++){
			phi_zw_test->me[i][j]=phi_xu_test->me[i][j]*phi_yv_test->me[i][j];
		}
	}
	double a=pow(sqrt(2*M_PI)*model->sigma,dy);
	MAT *tmp=sm_mlt(a,m_transp(model->coefficient,NULL),NULL);
	MAT *normalization=m_mlt(tmp,phi_xu_test,NULL);
	MAT *tmpB=m_mlt(m_transp(model->coefficient,NULL),phi_zw_test,NULL);
	
	MAT *ph=m_pixel(tmpB,normalization,NULL,'/');
	m_free(x);
	m_free(y);
	m_free(phi_xu_test);
	m_free(phi_yv_test);
	m_free(phi_zw_test);
	m_free(tmp);
	m_free(normalization);
	m_free(tmpB);
	return ph;
}

