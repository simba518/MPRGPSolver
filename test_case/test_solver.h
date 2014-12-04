#ifndef _TEST_SOLVER_H_
#define _TEST_SOLVER_H_

#include <MPRGPSolver.h>
#include <iostream>
using namespace std;
using namespace MATH;

typedef Eigen::Matrix<double,4,1> Vec4d;
typedef vector<Vec4d,Eigen::aligned_allocator<Vec4d> > VVec4d;
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

void testMPRGPPlaneSolver3D(){
  
  cout << "testMPRGPPlaneSolver3D " << endl;

  MatrixXd M(3,3);
  M << 1,0,0,
	0,1,0,
	0,0,1;
  const SparseMatrix<double> A = createFromDense(M);
  VectorXd b(3);
  VectorXd x(3);
  b << 0,0,0;
  x << 2,2,2;

  vector<Vector4d,aligned_allocator<Vector4d> > planes;
  Vector4d p;
  p << 1,0,0,-1;
  planes.push_back(p);

  p << 0,1,0,-2;
  planes.push_back(p);

  p << 0,0,1,3;
  planes.push_back(p);

  const int rlst_code = MPRGPPlane<double>::solve(FixedSparseMatrix<double>(A),b,planes,x);
  assert_eq(rlst_code,0);
  assert_eq(x[0],1);
  assert_eq(x[1],2);
  assert_eq(x[2],0);
}

void testMPRGPPlaneSolver3D_OnePlane(){
  
  cout << "testMPRGPPlaneSolver3D_OnePlane " << endl;

  MatrixXd M(3,3);
  M << 0.5f,0,0,
	0,0.5f,0,
	0,0,0.5f;
  const SparseMatrix<double> A = createFromDense(M);
  VectorXd b(3);
  VectorXd x(3);
  b << -1,-1,0;
  x << 0,0,0;

  vector<Vector4d,aligned_allocator<Vector4d> > planes;
  Vector4d p;
  p << 1,1,0,sqrt(2)/2;
  p.head(3) = p.head(3)/p.head(3).norm();
  planes.push_back(p);

  const int rlst_code = MPRGPPlane<double>::solve(FixedSparseMatrix<double>(A),b,planes,x);
  VectorXd correct_x(3);
  correct_x << -0.5,-0.5,0.0;
  assert_le((x-correct_x).norm(),1e-12);
  assert_eq(rlst_code,0);

  // test save and load then solve.
  VectorXd x2(x.size());
  x2.setZero();
  assert(writeQP(A,b,planes,x2,"tempt_test_io.mat"));
  const int c = MPRGPPlane<double>::solve("tempt_test_io.mat",x2);
  assert_le((x2-x).norm(),1e-10);
  assert_eq(c,0);
}

void testSolverFromFile(){

  cout << "testSolverFromFile " << endl;

  const string dir = "./test_case/data/";
  VectorXd x;
  int rlst_code = MPRGPPlane<double>::solve(dir+"one_tet_vp.QP",x,1e-3,100);
  assert_eq_ext(rlst_code,0,dir+"one_tet_vp.QP");

  rlst_code = MPRGPPlane<double>::solve(dir+"one_tet_vp2.QP",x,1e-3,100);
  assert_eq_ext(rlst_code,0,dir+"one_tet_vp2.QP");

  rlst_code = MPRGPPlane<double>::solve(dir+"one_tet_cone10.QP",x,1e-4,100);
  assert_eq_ext(rlst_code,0,dir+"one_tet_cone10.QP");

  rlst_code = MPRGPPlane<double>::solve(dir+"one_tet_ball.QP",x,1e-4,100);
  assert_eq_ext(rlst_code,0,dir+"one_tet_ball.QP");
}

void testComputeLagMultipliers(const string &QP_file, const double tol){

  Eigen::SparseMatrix<double> A;
  Eigen::Matrix<double,-1,1> B;
  VVec4d planes;
  VectorXd x;
  assert( loadQP(A, B, planes, x, QP_file) );

  {
	const VVec4d tempt = planes;
	planes.clear();
	for (int i = 0; i < tempt.size(); ++i){
	  const Vector3d ni = planes[i].segment<3>(0);
	  assert_in(ni.norm(), 1-1e-8, 1+1e-8);
	  int j = 0;
	  for ( ;j < planes.size(); ++j ){
		const Vector3d nj = planes[j].segment<3>(0);
		if((ni-nj).norm() < 1e-4)
		  break;
	  }
	  if( j == planes.size() )
		planes.push_back(tempt[i]);
	}
	// cout << "\n\n";
	// const MatrixXd M = A;
	// IOFormat OctaveFmt(StreamPrecision, 0, ", ", ";\n", "", "", "[", "]");
	// cout<< "\n\n" << setprecision(16) << M.format(OctaveFmt) << "\n\n";
  }

  VVVec4d planes_for_each_node;
  PlaneProjector<double>::convert(planes, planes_for_each_node, x.size()/3);

  PlaneProjector<double> projector(planes_for_each_node, x);
  MPRGPPlane<double>::solve(FixedSparseMatrix<double>(A),B,projector,x,tol,2000);

  const VectorXd g = A*x-B;
  const vector<vector<int> > &face_indices = projector.getFaceIndex();
  vector<vector<double> > all_lambdas;
  MPRGPPlane<double>::computeLagMultipliers(g,planes_for_each_node,face_indices,all_lambdas);

  VectorXd diff = g;
  assert_eq(planes_for_each_node.size(), all_lambdas.size());
  for (size_t i = 0; i < all_lambdas.size(); ++i){
	
	const vector<double> &lambdas = all_lambdas[i];
	const VVec4d &planes = planes_for_each_node[i];
	assert_eq(lambdas.size(), planes.size());
    for (size_t p = 0; p < lambdas.size(); ++p){
	  const double la = lambdas[p];
	  const Vector3d n = planes[p].segment<3>(0);
	  assert_ge(la,0.0);
	  diff.segment<3>(i*3) -= la*n;
	}
  }

  cout<< "\ng:"<< g.norm()<< ", " << g.transpose() << "\n\n";
  cout<< "d:"<< diff.norm() << ", " << diff.transpose() << "\n\n";

}

void testComputeLagMultipliers(){

  cout << "testComputeLagMultipliers " << endl;

  const string dir = "./test_case/data/";

  // testComputeLagMultipliers(dir+"one_tet_vp.QP", 1e-6);
  // testComputeLagMultipliers(dir+"one_tet_vp2.QP", 1e-6);
  // testComputeLagMultipliers(dir+"one_tet_cone10.QP", 1e-6);
  testComputeLagMultipliers(dir+"one_tet_ball.QP", 1e-6);

}

#endif /* _TEST_SOLVER_H_ */
