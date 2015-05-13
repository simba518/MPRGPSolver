#ifndef _TEST_ICASOLVER_H_
#define _TEST_ICASOLVER_H_

#include "ICASolver.h"
#include "test_utility.h"
using namespace MATH;

void test_GaussSeidel(){

  //example from: http://en.wikipedia.org/wiki/Gauss%E2%80%93Seidel_method
  MatrixXd dB(2,2);
  dB << 16,3, 7,-11;
  const SparseMatrix<double> B = createFromDense(dB);

  GaussSeidel<false> GS(7, 1e-4);
  GS.reset(B);

  VectorXd c(2);
  c << 11, 13;

  VectorXd lambda(2);
  lambda << 1, 1;

  const bool succ = GS.solve(c, lambda);
  assert(succ);

  GS.printSolveInfo(B, c, lambda);

  assert_le( abs(lambda[0] - 0.812190291031091), 1e-4 );
  assert_le( abs(lambda[1] - (-0.6649698147983967)), 1e-4 );

}

void test_ProjectedGaussSeidel(){
  
  const int n = 100;
  MatrixXd dB = MatrixXd::Random(n,n);
  dB = dB.transpose()*dB;
  dB += MatrixXd::Identity(n,n)*2;
  const SparseMatrix<double> B = createFromDense(dB);

  ProjectedGaussSeidel PGS(1000, 1e-8);
  PGS.reset(B);
  const VectorXd c = VectorXd::Random(n);
  VectorXd lambda(n);
  lambda.setZero();

  const bool succ = PGS.solve(c, lambda);
  PGS.printSolveInfo(B, c, lambda);
  assert(succ);

  assert_le(PGS.getIterations(), 47);
  assert_le(PGS.getResidual(), 1e-8);

  const VectorXd y = B*lambda-c;
  for (int i = 0; i < lambda.size(); ++i){
    assert_ge(lambda[i], -1e-7);
    assert_ge(y[i], -1e-7);
  }
  assert_le( abs(lambda.dot(y)), 1e-7 );
}

void test_ICASolver(){
  
  const int n = 4*3;
  MatrixXd dA = MatrixXd::Random(n,n);
  dA = dA.transpose()*dA;
  dA += MatrixXd::Identity(n,n)*3;
  const SparseMatrix<double> A = createFromDense(dA);

  const int m = n/2;
  const MatrixXd dJ = MatrixXd::Random(m,n);
  const SparseMatrix<double> J = createFromDense(dJ);

  const double tol = 1e-10;
  ICASolver ICA(100000, tol);
  ICA.reset(A);

  const VectorXd p = VectorXd::Random(m);
  VectorXd x = VectorXd::Random(n);
  bool succ = ICA.solve(J, p, x);
  ICA.printSolveInfo(A,J,p,x);
  assert(succ);

  assert_le(ICA.getResidual(), tol);
  assert_le(ICA.getIterations(), 22);
  assert_le((A*x - J.transpose()*ICA.getLambda()).norm(), tol);
  assert_le( abs(ICA.getLambda().dot(J*x-p)), tol );

  const VectorXd y = J*x-p;
  for (int i = 0; i < ICA.getLambda().size(); ++i){
    assert_ge(ICA.getLambda()[i], 0.0f);
  	assert_ge(y[i], -(tol*n));
  }
}

#endif /* _TEST_ICASOLVER_H_ */
