#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "matrix.h"
#include "mesch12b/matrix.h"
#include "mesch12b/matrix2.h"
#include "lscde.h"

#define Malloc(type,n) (type *)malloc((n)*sizeof(type))
#define Calloc(type,n) (type *)calloc(1,(n)*sizeof(type))

void printMatrix(MAT *a,char *name){
	int i,j;
	for(i=0;i<a->m;i++){
		for(j=0;j<a->n;j++){
			printf("%s[%d][%d]:%lf\n",name,i,j,a->me[i][j]);
		}
	}
}
MAT *m_pixel(MAT *a,MAT *b,MAT *out,char command){
	int i,j;
	if(a->m!=b->m || a->n!=b->n){
		fprintf(stderr,"don't match size\n");
		exit(1);
	}
	if(!out) out=m_get(a->m,a->n);
	for(i=0;i<a->m;i++){
		for(j=0;j<a->n;j++){
			switch(command){
			case '*':
				out->me[i][j]=a->me[i][j]*b->me[i][j];
				break;
			case '/':
				out->me[i][j]=a->me[i][j]*b->me[i][j];
				break;
			}
		}
	}
	return out;
}
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
MAT *getDistance(MAT *train,MAT *center,MAT *repmatTrain,MAT *repmatCenter){
	MAT *uTrans=m_transp(center,NULL);
	MAT *uTrans2x=sm_mlt(2.0,uTrans,NULL);
	MAT *ux2x=m_mlt(uTrans2x,train,NULL);
	MAT *tmp1=m_add(repmatTrain,repmatCenter,NULL);

	m_free(uTrans);
	m_free(uTrans2x);
	/*m_free(ux2x);
	m_free(tmp1);*/
	return m_sub(tmp1,ux2x,NULL);
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
MAT *makeLearning(double sigma,double lambda,int b,int numSample){
	int i,j;
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
	MAT *sum=m_get(b,b);
	m_zero(sum);
	for(i=0;i<fold;i++){
		if(i!=k){
			m_add(sum,Phibar_cv[i],sum);
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
	return mylinsolve(matD,matF);
}
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

	double *sigma_list=logspace(-1.5,1.5,9);
	double sigma;
	double *lambda_list=logspace(-3.0,1.0,9);
	double lambda;
	double cc=2.2204e-16;


	MAT *xCenters=m_get(1,b);
	MAT *yCenters=m_get(1,b);
	MAT *tmp1NX=m_get(1,numSample);
	MAT *tmp1NY=m_get(1,numSample);
	MAT *tmp1bU=m_get(1,b);
	MAT *tmp1bV=m_get(1,b);
	MAT *repmatXX=m_get(b,numSample);
	MAT *repmatYY=m_get(b,numSample);
	MAT *repmatUU=m_get(b,numSample);
	MAT *repmatVV=m_get(b,numSample);
	MAT *repmatV2=m_get(b,b);
	MAT *repmatV2T=m_get(b,b);
	MAT *x_train=m_get(1,numSample);
	MAT *y_train=m_get(1,numSample);
	MAT *alphat;
	MAT **Phibar_cv=malloc(fold*sizeof(MAT *));

	randperm(randIndex,numSample);
	randperm(cv_index,numSample);

	for(i=0;i<b;i++){
		xCenters->me[0][i]=xTrain[randIndex[i]];
		yCenters->me[0][i]=yTrain[randIndex[i]];
		tmp1bU->me[0][i]=xCenters->me[0][i]*xCenters->me[0][i];
		tmp1bV->me[0][i]=yCenters->me[0][i]*yCenters->me[0][i];
	}
	for(i=0;i<numSample;i++){
		x_train->me[0][i]=xTrain[i];
		y_train->me[0][i]=yTrain[i];
		tmp1NX->me[0][i]=xTrain[i]*xTrain[i];
		tmp1NY->me[0][i]=yTrain[i]*yTrain[i];
	}

	for(i=0;i<b;i++){
		for(j=0;j<numSample;j++){
			repmatXX->me[i][j]=tmp1NX->me[0][j];
			repmatYY->me[i][j]=tmp1NY->me[0][j];
			repmatUU->me[i][j]=tmp1bU->me[0][i];
			repmatVV->me[i][j]=tmp1bV->me[0][i];
		}
	}
	MAT *xu_dist2=getDistance(x_train,xCenters,repmatXX,repmatUU);
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

		}
		for(j=0;j<9;j++){
			lambda=lambda_list[j];
			double score_tmp=0.0;
			for(k=0;k<fold;k++){
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
				MAT *matC=sm_mlt(lambda,matB,NULL);
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
				for(i=0;i<b;i++){
						alphat->me[i][0]=(alphat->me[i][0] < 0 ) ? 0: alphat->me[i][0];
				}
				MAT *tmp=m_get(phi_xu->m,numSample/fold+1);

				for(k=0;k<phi_xu->m;k++){
					int tmpCt=0;
					for(l=0;l<phi_xu->n;l++){
						if(cv_split[l]==j){
							tmp->me[k][tmpCt++]=phi_xu->me[k][l];
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
		}*/

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
		m_free(phi_zwMean);*/
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
	double fsigma=sigma_list[sigma_index];
	double flambda=lambda_list[idx[sigma_index]];

			

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

	return model;
}

/*
	compute P(Y|X)
*/
MAT *kernel_Gaussian(MAT *x,MAT *center,double sigma){
	int i,j;
	MAT *center2=m_pixel(center,center,NULL,'*');
	MAT *x2=m_pixel(x,x,NULL,'*');
	MAT *tmpA=m_get(x->n,center->n);
	MAT *tmpB=m_get(x->n,center->n);
	for(i=0;i<x->n;i++){
		for(j=0;j<center->n;j++){
			tmpA->me[i][j]=center2->me[0][j];
			tmpB->me[i][j]=x2->me[0][j];
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
	//printMatrix(phi_xu_test,"phi_xu_test");
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

