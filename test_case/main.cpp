#include <MPRGPSolver.h>
using namespace MATH;

int main(int argc, char *argv[]){

  const int n = 10;
  SparseMatrix<double> A(n,n);
  const VectorXd b = VectorXd::Random(n);
  VectorXd L(n), x(n);
  MPRGPLowerBound<double>::solve(FixedSparseMatrix<double>(A),b,L,x);

  cout<< "residual: " << (A*x-b).norm() << endl;
  
  return 0;
}
