#ifndef _TEST_DECOUPLED_MPRGP_H_
#define _TEST_DECOUPLED_MPRGP_H_

#include <MPRGPSolver.h>
#include "test_utility.h"
using namespace MATH;

void test_DecoupledMprgp(){
  
  cout << "testDecoupledMprgp " << endl;

  // init QP 
  const double tol = 1e-6;
  const int max_it = 10;
  const int n = 2;
  const MatrixXd M = MatrixXd::Random(n,n) + MatrixXd::Identity(n,n)*3.0f;
  const MatrixXd MtM = M.transpose()*M;
  const SparseMatrix<double> A = createFromDense(MtM);

  const MatrixXd JM = MatrixXd::Identity(std::min<int>(1,n),n)*3;
  const SparseMatrix<double> J = createFromDense(JM);

  const VectorXd b = VectorXd::Random(n);
  const VectorXd c = VectorXd::Random(J.rows());

  {// check matrices
	SelfAdjointEigenSolver<MatrixXd> es(A);
	cout << "eige(A): " << es.eigenvalues().transpose() << endl;
	// cout << "J:\n" << MatrixXd(J) << endl;
  }

  // init projector
  const SparseMatrix<double> JJt_mat = J*J.transpose();
  assert_eq_ext(JJt_mat.nonZeros(), J.rows(), "Matrix J is not decoupled.\n" << J);
  VectorXd JJt;
  MATH::getDiagonal(JJt_mat, JJt);
  DecoupledConProjector<double> projector(J, JJt, c);

  // get init value
  VectorXd y(n), x(n);
  y.setZero();
  projector.project(y, x);
  assert_eq(x.size(), y.size());
  assert_ext(projector.isFeasible(x), x.transpose());

  // solve
  typedef FixedSparseMatrix<double> MAT;
  MAT FA(A);
  MPRGPDecoupledCon<double>::solve<MAT,false>(FA, b, projector, x, tol, max_it);
  
}

void test_MPRGPLowerBound(){
  
  // init QP 
  const double tol = 1e-5;
  const int max_it = 1000;
  const int n = 10;
  const MatrixXd M = MatrixXd::Random(n,n) + MatrixXd::Identity(n,n)*10.0f;
  const MatrixXd MtM = M.transpose()*M;
  const SparseMatrix<double> A = createFromDense(MtM);

  const VectorXd b = VectorXd::Random(n);
  const VectorXd c = VectorXd::Random(n);

  {// check matrices
	SelfAdjointEigenSolver<MatrixXd> es(A);
	cout << "eige(A): " << es.eigenvalues().transpose() << endl;
  }

  VectorXd x = c;
  const FixedSparseMatrix<double> FA(A);
  MPRGPLowerBound<double>::solve(FA, b, c, x, tol, max_it);
}

#endif /* _TEST_DECOUPLED_MPRGP_H_ */
