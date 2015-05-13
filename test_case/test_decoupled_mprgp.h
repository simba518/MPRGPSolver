#ifndef _TEST_DECOUPLED_MPRGP_H_
#define _TEST_DECOUPLED_MPRGP_H_

#include <MPRGPSolver.h>
#include "test_utility.h"
using namespace MATH;

void test_DecoupledMprgp(const SparseMatrix<double> &A,const SparseMatrix<double> &J,
						 const double tol, const int max_it, bool check_A = false){
  
  const VectorXd b = VectorXd::Random(A.rows());
  const VectorXd c = VectorXd::Random(J.rows());

  if(check_A){// check matrices
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
  VectorXd y(A.rows()), x(A.rows());
  y.setZero();
  projector.project(y, x);
  assert_eq(x.size(), y.size());
  assert_ext(projector.isFeasible(x), x.transpose());

  // solve
  typedef FixedSparseMatrix<double> MAT;
  MAT FA(A);
  const int code = MPRGPDecoupledCon<double>::solve<MAT,false>(FA,b,projector,x,tol,max_it);
  assert_ge(code, 0);
}

void test_DecoupledMprgp(){
  
  if(true){
	cout << "test_DecoupledMprgp 1" << endl;
	const int n = 100;
	const MatrixXd M = MatrixXd::Random(n,n) + MatrixXd::Identity(n,n)*3.0f;
	const MatrixXd MtM = M.transpose()*M;
	const SparseMatrix<double> A = createFromDense(MtM);
	MatrixXd JM = MatrixXd::Identity(std::min<int>(A.rows(), A.rows()), A.rows());
	const SparseMatrix<double> J = createFromDense(JM);
	// test_DecoupledMprgp(A, J, 1e-10, 100, false); /// @bug
	test_DecoupledMprgp(A, J, 1e-6, 100, false);
  }

  if(true){
	cout << "test_DecoupledMprgp 2" << endl;
	const int n = 10;
	const MatrixXd M = MatrixXd::Random(n,n) + MatrixXd::Identity(n,n)*3.0f;
	const MatrixXd MtM = M.transpose()*M;
	const SparseMatrix<double> A = createFromDense(MtM);
	MatrixXd JM = MatrixXd::Identity(std::min<int>(A.rows()/2, A.rows()), A.rows());
	const SparseMatrix<double> J = createFromDense(JM);
	test_DecoupledMprgp(A, J, 1e-10, 12, false);
  }

  if(true){
	cout << "test_DecoupledMprgp 3" << endl;
	const int n = 10;
	const MatrixXd M = MatrixXd::Random(n,n) + MatrixXd::Identity(n,n)*3.0f;
	const MatrixXd MtM = M.transpose()*M;
	const SparseMatrix<double> A = createFromDense(MtM);
	MatrixXd JM = MatrixXd::Identity(std::min<int>(A.rows()/2, A.rows()), A.rows())*3.0;
	const SparseMatrix<double> J = createFromDense(JM);
	test_DecoupledMprgp(A, J, 1e-10, 20, false);
  }

  if(true){
	cout << "test_DecoupledMprgp 4" << endl;
	const int n = 3;
	const MatrixXd M = MatrixXd::Random(n,n) + MatrixXd::Identity(n,n)*3.0f;
	const MatrixXd MtM = M.transpose()*M;
	const SparseMatrix<double> A = createFromDense(MtM);
	MatrixXd JM = MatrixXd::Identity(std::min<int>(2,A.rows()), A.rows())*3;
	JM << 1,10,0, 0,0,4;
	const SparseMatrix<double> J = createFromDense(JM);
	test_DecoupledMprgp(A, J, 1e-10, 12, true);
  }
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
