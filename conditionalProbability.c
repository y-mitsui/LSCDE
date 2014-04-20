#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "matrix.h"
#include "mystring.h"
#include "mesch12b/matrix.h"
#include "mesch12b/matrix2.h"
#include "lscde.h"

#define Malloc(type,n) (type *)malloc((n)*sizeof(type))
#define Calloc(type,n) (type *)calloc(1,(n)*sizeof(type))

int main(void){
	FILE *fp;
	double *xSample=Malloc(double,NUM_SAMPLE);
	double *ySample=Malloc(double,NUM_SAMPLE);
	double *xTest=Malloc(double,NUM_TEST);
	double *yTest=Malloc(double,NUM_TEST);
	char buf[1024];
	int i;

	if(!(fp=fopen("normalX.csv","r"))){
		perror("normalX.csv");
		return 1;
	}
	i=0;
	while(fgets(buf,sizeof(buf),fp)){
		xSample[i]=atof(buf);
		i++;
	}
	fclose(fp);

	if(!(fp=fopen("normalY.csv","r"))){
		perror("normalY.csv");
		return 1;
	}
	i=0;
	while(fgets(buf,sizeof(buf),fp)){
		ySample[i]=atof(buf);
		i++;
	}
	fclose(fp);

	lscde *ctx=lscdeModel(xSample,ySample,1,1,i);

	if(!(fp=fopen("testX.csv","r"))){
		perror("testX.csv");
		return 1;
	}
	i=0;
	while(fgets(buf,sizeof(buf),fp)){
		xTest[i]=atof(buf);
		i++;
	}
	fclose(fp);

	if(!(fp=fopen("testY.csv","r"))){
		perror("testY.csv");
		return 1;
	}
	i=0;
	while(fgets(buf,sizeof(buf),fp)){
		yTest[i]=atof(buf);
		i++;
	}
	fclose(fp);

	lscdeConditionalDensity(ctx,xTest,yTest,i);

	free(xSample);
	free(ySample);
	free(xTest);
	free(yTest);
	free(ctx);
	return 0;
}
