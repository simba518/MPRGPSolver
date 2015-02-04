#ifndef _TEST_UTILITY_H_
#define _TEST_UTILITY_H_

#include "MPRGPUtility.h"
#include <iostream>
using namespace std;
using namespace Eigen;
using namespace MATH;

typedef Matrix<double,4,1> Vec4d;
typedef vector<Vec4d,aligned_allocator<Vec4d> > VVec4d;
typedef vector<VVec4d > VVVec4d;

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

typedef vector<Matrix<double,4,1>, aligned_allocator<Matrix<double,4,1> > > VVec4Xd;
typedef vector<VVec4Xd > VVVec4Xd;

void test_io(){

  cout << "test io" << endl;
  
  const string file_name = "./tempt_test_io.mat";

  const int nodes = 10;
  const int n = nodes*3;
  const int p = 2;
  SparseMatrix<double> A,A2;
  VectorXd B,B2;
  VectorXd x,x2;
  VVec4Xd planes(p), planes2;

  const MatrixXd M = MatrixXd::Random(n,n);
  A = createFromDense(M);
  B = VectorXd::Random(n);
  x = VectorXd::Random(n);
  for (int i = 0; i < p; ++i)
    planes[i]<<1,1,0,i;

  {
	const bool test_write = writeQP<double>(A,B,planes,x,file_name);
	assert(test_write);

	const bool test_load = loadQP<double>(A2,B2,planes2,x2,file_name);
	assert(test_load);
  
	assert_eq(A2.size(), A.size());
	assert_le((A-A2).norm(),1e-11);

	assert_eq(B2.size(), B.size());
	assert_le((B-B2).norm(),1e-11);

	assert_eq(x2.size(), x.size());
	assert_le((x-x2).norm(),1e-11);

	assert_eq(planes2.size(), planes.size());
	for (int i = 0; i < (int)planes2.size(); ++i)
	  assert_le((planes2[i]-planes[i]).norm(),1e-11);
  }

  {
	VVVec4Xd planes_for_each_node, planes2_for_each_node;
	convert<double>(planes, planes_for_each_node, nodes);

	const bool test_write = writeQP<double>(A,B,planes_for_each_node,x,file_name);
	assert(test_write);

	A2.setZero();
	B2.setZero();
	x2.setZero();
	const bool test_load = loadQP<double>(A2,B2,planes2_for_each_node,x2,file_name);
	assert(test_load);
	
	assert_eq(A2.size(), A.size());
	assert_le((A-A2).norm(),1e-11);

	assert_eq(B2.size(), B.size());
	assert_le((B-B2).norm(),1e-11);

	assert_eq(x2.size(), x.size());
	assert_le((x-x2).norm(),1e-11);

	assert_eq(planes2_for_each_node.size(), planes_for_each_node.size());
	for (int i = 0; i < (int)planes2_for_each_node.size(); ++i){
	  for (int j = 0; j < (int)planes2_for_each_node[i].size(); ++j){
		assert_le((planes2_for_each_node[i][j]-planes2_for_each_node[i][j]).norm(),1e-11);
	  }
	}
  }
  
}

void test_io2(){

  cout << "test io2" << endl;

  const int n = 5;
  const int m = 2;
  const MatrixXd Am = MatrixXd::Random(n,n);
  const MatrixXd Jm = MatrixXd::Random(m,n);
  const Eigen::SparseMatrix<double> A = createFromDense(Am);
  const Eigen::SparseMatrix<double> J = createFromDense(Jm);
  const Eigen::VectorXd B = VectorXd::Random(A.rows());
  const Eigen::VectorXd x = VectorXd::Random(A.rows());
  const Eigen::VectorXd c = VectorXd::Random(J.rows());

  const string fname = "./tempt_test_io.mat";
  const bool succ_write = writeQP(A, B, J, c, x, fname);
  assert_ext(succ_write, fname);

  SparseMatrix<double> A2, J2;
  VectorXd B2, c2, x2;
  const bool succ_load = loadQP(A2, B2, J2, c2, x2, fname);
  assert_ext(succ_load, fname);

  assert_eq(x, x2);
  assert_eq(B, B2);
  assert_eq(c, c2);
  assert_eq((J-J2).norm(), 0);
  assert_eq((A-A2).norm(), 0);

  const string project_dir = "/home/simba/Workspace/CollisionHandle/";
  const string qp = project_dir+"/data/cube/tempt_2cube_mprgp14000/QP/qp6.b";
  const bool load_succ = loadQP(A2, B2, J2, c2, x2, qp);
  assert(load_succ);

}

#endif /* _TEST_UTILITY_H_ */
