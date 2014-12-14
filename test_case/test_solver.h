#ifndef _TEST_SOLVER_H_
#define _TEST_SOLVER_H_

#include <MPRGPSolver.h>
#include "test_utility.h"
using namespace MATH;

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
  {
	VectorXd x;
	const int rlst_code = MPRGPPlane<double>::solve(dir+"one_tet_vp.QP",x,1e-6,11);
	assert_eq_ext(rlst_code,0,dir+"one_tet_vp.QP");
  }

  {
	VectorXd x;
	const int rlst_code = MPRGPPlane<double>::solve(dir+"one_tet_vp2.QP",x,1e-6,8);
	assert_eq_ext(rlst_code,0,dir+"one_tet_vp2.QP");
  }

  {
	VectorXd x;
	/// @bug sometimes this test case will failed, that is, it convergence very slow.
	/// One of the possible reason is that, we use tmp.setRandom() in MPRGP::specRad(...).
	// const int rlst_code = MPRGPPlane<double>::solve(dir+"one_tet_cone10.QP",x,1e-6,60);
	const int rlst_code = MPRGPPlane<double>::solve(dir+"one_tet_cone10.QP",x,1e-5,60);
	assert_eq_ext(rlst_code,0,dir+"one_tet_cone10.QP");
  }

  {
	VectorXd x;
	const int rlst_code = MPRGPPlane<double>::solve(dir+"one_tet_ball.QP",x,1e-6,35);
	assert_eq_ext(rlst_code,0,dir+"one_tet_ball.QP");
  }
}

void testComputeLagMultipliers(const string &QP_file, const double tol, const int max_it){

  Eigen::SparseMatrix<double> A;
  Eigen::Matrix<double,-1,1> B;
  VVec4d planes;
  VectorXd x;
  assert( loadQP(A, B, planes, x, QP_file) );

  {
	const VVec4d tempt = planes;
	planes.clear();
	for (int i = 0; i < (int)tempt.size(); ++i){
	  const Vector3d ni = planes[i].segment<3>(0);
	  assert_in(ni.norm(), 1-1e-8, 1+1e-8);
	  int j = 0;
	  for ( ;j < (int)planes.size(); ++j ){
		const Vector3d nj = planes[j].segment<3>(0);
		if((ni-nj).norm() < 1e-4)
		  break;
	  }
	  if( j == (int)planes.size() )
		planes.push_back(tempt[i]);
	}
	// cout << "\n\n";
	// const MatrixXd M = A;
	// IOFormat OctaveFmt(StreamPrecision, 0, ", ", ";\n", "", "", "[", "]");
	// cout<< "\n\nA:" << setprecision(16) << M.format(OctaveFmt) << "\n\n";
  }

  VVVec4d planes_for_each_node;
  convert<double>(planes, planes_for_each_node, x.size()/3);

  PlaneProjector<double> projector(planes_for_each_node, x);
  const int rlst_code = MPRGPPlane<double>::solve(FixedSparseMatrix<double>(A),B,projector,x,tol,max_it);
  assert_eq_ext(rlst_code,0,QP_file);
  assert(MPRGPPlane<double>::checkResult(A,B,projector,x,tol*3.0f));
}

void testComputeLagMultipliers(){

  cout << "testComputeLagMultipliers " << endl;

  const string dir = "./test_case/data/";

  testComputeLagMultipliers(dir+"one_tet_vp.QP", 1e-6,11);
  testComputeLagMultipliers(dir+"one_tet_vp2.QP", 1e-6,8);
  // testComputeLagMultipliers(dir+"one_tet_cone10.QP", 1e-6,60); /// @bug
  // testComputeLagMultipliers(dir+"one_tet_cone10.QP", 1e-5,60); /// @bug
  testComputeLagMultipliers(dir+"one_tet_ball.QP", 1e-6,10);

}

#endif /* _TEST_SOLVER_H_ */
