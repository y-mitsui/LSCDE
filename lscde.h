
typedef struct{
	MAT *coefficient;
	MAT *xCenters;
	MAT *yCenters;
	double sigma;
}lscde;
lscde* lscdeModel(double *xTrain,double *yTrain,int xDim,int yDim,int numSample);
MAT* lscdeConditionalDensity(lscde *model,double *X,double *Y,int numTest);

FILE *dfopen(char *path,char *mode);
void fload(char* path,int *data);
void floadD(char* path,double *data);

