#ifndef _TEST_PRECONDITION_H_
#define _TEST_PRECONDITION_H_

#include <MPRGPPrecondition.h>
#include "test_utility.h"
using namespace MATH;

void test_tridiagonalPrecond(){
  
  cout << "test_tridiagonalPrecond\n";

  typedef TridiagonalPlanePreconSolver<double,FixedSparseMatrix<double> > Preconditioner;
  const MatrixXd Md = MatrixXd::Random(4,4);
  const SparseMatrix<double> M = createFromDense(Md);
  SparseMatrix<double> T;
  Preconditioner::buildTridiagonalMatrix(M,T);

  MatrixXd Td = T;
  assert_eq(Td(0,2), 0);
  assert_eq(Td(0,3), 0);
  assert_eq(Td(1,3), 0);

  assert_eq(Td(2,0), 0);
  assert_eq(Td(3,0), 0);
  assert_eq(Td(3,1), 0);

  Td(0,2) = Md(0,2);
  Td(0,3) = Md(0,3);
  Td(1,3) = Md(1,3);
  Td(2,0) = Md(2,0);
  Td(3,0) = Md(3,0);
  Td(3,1) = Md(3,1);

  assert_eq(Td, Md);
}

#endif /* _TEST_PRECONDITION_H_ */
