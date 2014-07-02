#include <MPRGPSolver.h>
using namespace MATH;

#include <float.h>
float ScalarUtil<float>::scalar_max=FLT_MAX;
float ScalarUtil<float>::scalar_eps=1E-5f;
double ScalarUtil<double>::scalar_max=DBL_MAX;
double ScalarUtil<double>::scalar_eps=1E-9;

int main(int argc, char *argv[]){

  const int n = 10;
  SparseMatrix<double> A(n,n);
  const VectorXd b = VectorXd::Random(n);
  VectorXd L(n), x(n);
  MPRGPLowerBound<double>::solve(FixedSparseMatrix<double>(A),b,L,x);

  cout<< "residual: " << (A*x-b).norm() << endl;
  
  return 0;
}
