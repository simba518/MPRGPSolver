#ifndef _TEST_SOLVER_H_
#define _TEST_SOLVER_H_

#include <MPRGPSolver.h>
using namespace MATH;

template <class T>
const SparseMatrix<T> &createFromDense(const Matrix<T,-1,-1> &M, SparseMatrix<T> &S, const T tol=1e-16){
  
  typedef Triplet<T> E_Triplet;
  std::vector<E_Triplet> striplet;
  striplet.reserve(M.size());
  for (int i = 0; i < M.rows(); ++i) {
	for (int j = 0; j < M.cols(); ++j) {
	  if ( fabs(M(i,j)) >= tol )
		striplet.push_back( E_Triplet(i,j,M(i,j)) );
	}
  }
  S.resize(M.rows(), M.cols());
  S.setFromTriplets(striplet.begin(), striplet.end());
  return S;
}

template <class T> 
const SparseMatrix<T> createFromDense(const Matrix<T,-1,-1> &M, const T tol=1e-16){
  SparseMatrix<T> S;
  return createFromDense(M,S,tol);
}

void test1DSolverLB(){

  cout << "test 1d with x >= L" << endl;

  MatrixXd M(1,1);
  M << 2;
  const SparseMatrix<double> A = createFromDense(M);
  VectorXd b(1);
  VectorXd L(1), x(1);
  b << -1;
  L << 1;
  x << 2;

  const int rlst_code = MPRGPLowerBound<double>::solve(FixedSparseMatrix<double>(A),b,L,x);
  assert_eq(rlst_code,0);
  assert_eq(x[0],1);
}

void test2DSolverLB(){

  cout << "test 2d with x >= L" << endl;

  MatrixXd M(2,2);
  M << 2,0,0,2;
  const SparseMatrix<double> A = createFromDense(M);
  VectorXd b(2);
  VectorXd L(2), x(2);
  b << -1,-1;
  L << 1,2;
  x << 2,2;

  const int rlst_code = MPRGPLowerBound<double>::solve(FixedSparseMatrix<double>(A),b,L,x);
  assert_eq(rlst_code,0);
  assert_eq(x[0],1);
  assert_eq(x[1],2);
}

void test1DSolverBB(){

  cout << "test 1d with H >= x >= L" << endl;

  MatrixXd M(1,1);
  M << 2;
  const SparseMatrix<double> A = createFromDense(M);
  VectorXd b(1);
  VectorXd L(1), x(1), H(1);
  b << -1;
  L << 1;
  H << 2.1;
  x << 2;

  const int rlst_code = MPRGPBoxBound<double>::solve(FixedSparseMatrix<double>(A),b,L,H,x);
  assert_eq(rlst_code,0);
  assert_eq(x[0],1);

  L << -5;
  H << -2;
  x << -3;
  const int rlst_code2 = MPRGPBoxBound<double>::solve(FixedSparseMatrix<double>(A),b,L,H,x);
  assert_eq(rlst_code2,0);
  assert_eq(x[0],-2);
}

void test2DSolverBB(){

  cout << "test 2d with H >= x >= L" << endl;

  MatrixXd M(2,2);
  M << 2,0,0,2;
  const SparseMatrix<double> A = createFromDense(M);
  VectorXd b(2);
  VectorXd L(2), x(2), H(2);
  b << -1,-1;
  L << 1,2;
  H << 3,2.1;
  x << 2,2;

  const int rlst_code = MPRGPBoxBound<double>::solve(FixedSparseMatrix<double>(A),b,L,H,x);
  assert_eq(rlst_code,0);
  assert_eq(x[0],1);
  assert_eq(x[1],2);


  L << -10,-20;
  H << -3,-2.1;
  x << -4,-4;

  const int rlst_code2 = MPRGPBoxBound<double>::solve(FixedSparseMatrix<double>(A),b,L,H,x);
  assert_eq(rlst_code2,0);
  assert_eq(x[0],-3);
  assert_eq(x[1],-2.1);

}

#endif /* _TEST_SOLVER_H_ */
