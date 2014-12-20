#ifndef _TEST_SIMULATION_H_
#define _TEST_SIMULATION_H_

#include <MPRGPSolver.h>
#include "test_utility.h"
using namespace MATH;

void Cholesky(const Matrix3d &M, Matrix3d &L){
  
  LLT<Matrix3d> lltOfM(M);
  L = lltOfM.matrixL();
  assert_le( (L*L.transpose()-M).norm(), 1e-9);
}

void SysGaussianShield(const Matrix3d &M, Matrix3d &L){
  
  L = M;
  L(0,1) = 0;
  L(0,2) = 0;
  L(1,2) = 0;
  assert_gt_ext(M(0,0) ,0, M);
  assert_gt_ext(M(1,1) ,0, M);
  assert_gt_ext(M(2,2) ,0, M);
  Vector3d d;
  d[0] = 1.0/sqrt(M(0,0));
  d[1] = 1.0/sqrt(M(1,1));
  d[2] = 1.0/sqrt(M(2,2));
  L *= d.asDiagonal();
}

void blockDiagPrecond(SparseMatrix<double> &A,VectorXd &B, VectorXd &x, VVVec4d &planes){

  vector<Matrix3d> d(B.size()/3);
  vector<Matrix3d> L=d, L_inv=d;
  for (int i = 0; i < (int)d.size(); ++i){
    d[i].setZero();
  }
  for (int k = 0; k < A.outerSize(); ++k){
	for(typename SparseMatrix<double>::InnerIterator it(A,k);it;++it){
	  const int r = it.row();
	  const int c = it.col();
	  if(r/3 == c/3){
		assert_eq(d[r/3](r%3,c%3), 0);
		d[r/3](r%3,c%3) = it.value();
	  }
	}
  }

  for (int i = 0; i < (int)d.size(); ++i){

	SysGaussianShield(d[i], L[i]);
	// Cholesky(d[i], L[i]);
	L_inv[i] = L[i].inverse();
	assert_eq(L_inv[i], L_inv[i]);
  }

  vector<Triplet<double> > triplet;
  triplet.reserve(L_inv.size()*9);
  for (int i = 0; i < (int)L_inv.size(); ++i){
	const Matrix3d &m = L_inv[i];
    for (int r = 0; r < 3; ++r){
	  for (int c = 0; c < 3; ++c){
		const int rr = i*3+r;
		const int cc = i*3+c;
		triplet.push_back(Triplet<double>(rr,cc, m(r,c)));
	  }
	}
  }
  SparseMatrix<double> ML_inv(B.size(), B.size());
  ML_inv.reserve(triplet.size());
  ML_inv.setFromTriplets(triplet.begin(), triplet.end());

  SparseMatrix<double> temp_A = A;
  A = ML_inv*temp_A*ML_inv.transpose();

  for (int i = 0; i < (int)d.size(); ++i){
    B.segment<3>(i*3) = L_inv[i]*B.segment<3>(i*3);
    x.segment<3>(i*3) = L[i].transpose()*x.segment<3>(i*3);
  }

  for (int vi = 0; vi < (int)planes.size(); ++vi){
	const Matrix3d &Li = L_inv[vi];
    for (int pi = 0; pi < (int)planes[vi].size(); ++pi){
	  const Vector3d n = Li*(planes[vi][pi].segment<3>(0));
	  const double inv_norm = 1.0f/n.norm();
	  assert_eq(inv_norm, inv_norm);
	  const double p = planes[vi][pi][3]*inv_norm;
	  planes[vi][pi].segment<3>(0) = (n*inv_norm);
	  planes[vi][pi][3] = p;
	}
  }
}

void diagPrecond(SparseMatrix<double> &A,VectorXd &B, VectorXd &x, VVVec4d &planes){

  VectorXd d(B.size());
  for(int k=0;k<A.outerSize();++k){
	for(typename SparseMatrix<double>::InnerIterator it(A,k);it;++it){
	  if (it.col() == it.row()){
		assert_eq(it.row(),k);
		d[k] = it.value();
		break;
	  }
	}
  }
  
  VectorXd L(d.size()), L_inv(d.size());
  for (int i = 0; i < d.size(); ++i){
	assert_gt(d[i], ScalarUtil<double>::scalar_eps);
    L[i] = sqrt(d[i]);
    L_inv[i] = 1.0f/L[i];
  }

  SparseMatrix<double> temp_A = A;
  A = L_inv.asDiagonal()*temp_A*L_inv.asDiagonal();
  for (int i = 0; i < L.size(); ++i){
    B[i] = B[i]*L_inv[i];
    x[i] = x[i]*L[i];
  }

  for (int vi = 0; vi < (int)planes.size(); ++vi){
	const Vector3d Li = L_inv.segment<3>(vi*3);
    for (int pi = 0; pi < (int)planes[vi].size(); ++pi){
	  const Vector3d n = Li.asDiagonal()*(planes[vi][pi].segment<3>(0));
	  const double inv_norm = 1.0f/n.norm();
	  assert_eq(inv_norm, inv_norm);
	  const double p = planes[vi][pi][3]*inv_norm;
	  planes[vi][pi].segment<3>(0) = (n*inv_norm);
	  planes[vi][pi][3] = p;
	}
  }
}

template <typename Preconditioner=DiagonalPlanePreconSolver<double,FixedSparseMatrix<double>, true> >
double testQPFromFile(const string &file_name,const double tol, const int max_it, const bool no_con=false){

  const string dir = "./test_case/data/";
  
  SparseMatrix<double> A;
  VectorXd B, x;
  VVVec4d planes_for_each_node;
  const bool succ_to_load_QP = loadQP(A, B, planes_for_each_node, x, dir+file_name);
  assert_ext(succ_to_load_QP, dir+file_name);
  cout << "dimension: " << B.size() << endl;
  if(no_con){
	const size_t n = planes_for_each_node.size();
	planes_for_each_node.clear();
	planes_for_each_node.resize(n);
  }

  // const double fun0 = (x.dot(A*x))*0.5f-x.dot(B);
  const double norm0 = (A*x-B).norm();

  // blockDiagPrecond(A, B, x, planes_for_each_node);
  // diagPrecond(A, B, x, planes_for_each_node);
  const double norm0_pre = (A*x-B).norm();
  const double pre_tol = tol*norm0_pre/norm0;
  cout<< "pre tol: " << pre_tol << endl;
  
  PlaneProjector<double> projector(planes_for_each_node, x);
  const FixedSparseMatrix<double> SA(A);
  const int code = MPRGPPlane<double>::solve<FixedSparseMatrix<double>, Preconditioner>(SA,B,projector,x,pre_tol,max_it);
  ERROR_LOG_COND("MPRGP is not convergent, result code is "<<code<<endl,code==0);
  DEBUG_FUN( MPRGPPlane<double>::checkResult(A, B, projector, x, tol) );
  assert( isFeasible(planes_for_each_node, x) );

  // x = L_inv.asDiagonal()*x;
  const double fun1 = (x.dot(A*x))*0.5f-x.dot(B);
  cout<< setprecision(12) << "function value: " << fun1 << endl;

  {
	// check norm and function value
	// SimplicialCholesky<SparseMatrix<double> > sol(A);
	// const VectorXd xx = sol.solve(B);
	// const double norm1 = (A*x-B).norm();
	// const double norm2 = (A*xx-B).norm();
	// const double fun2 = (xx.dot(A*xx))*0.5f-xx.dot(B);
	// assert_lt(fun1, fun0);
	// assert_le(fun2, fun1);
	// assert_lt(norm1, norm0); // @bug
	// assert_le(norm2, norm1);
  }

  return fun1;
}

bool loadAdiGroups(const string filename, vector<vector<set<int> > > &groups){

  ifstream in(filename.c_str());
  if( !in.is_open()){
	cout << "ERROR: failed to open: " << filename << endl;
	return false;
  }

  groups.clear();
  groups.resize(3);

  string tempt;
  int num_g = 0;
  int len = 0;
  int vid = 0;
  for (int direction = 0; direction < 3; ++direction){

	in >> tempt >> tempt >> tempt >> num_g;
	assert(num_g > 0);
	groups[direction].resize(num_g);
	for (int g = 0; g < num_g; ++g){
	  in >> tempt >> len;
	  assert(len >= 0);
	  for (int i = 0; i < len; ++i){
		in >> vid;
		groups[direction][g].insert(vid);
	  }
	}
  }

  in.close();
  return in.good();
}

void testADI(const string &file_name,const double tol, const int max_it, const bool no_con=false){

  const string dir = "./test_case/data/";
  
  SparseMatrix<double> A;
  VectorXd B, x;
  VVVec4d planes_for_each_node;
  const bool succ_to_load_QP = loadQP(A, B, planes_for_each_node, x, dir+file_name);
  assert_ext(succ_to_load_QP, dir+file_name);
  assert_eq(A.rows(), B.size());
  assert_eq(A.cols(), B.size());
  assert_eq(x.size(), B.size());
  cout << "dimension: " << B.size() << endl;
  if(no_con){
	const size_t n = planes_for_each_node.size();
	planes_for_each_node.clear();
	planes_for_each_node.resize(n);
  }

  vector<vector<set<int> > > groups;
  { // init groups
	// const int nodes = planes_for_each_node.size();
	// vector<set<int> > group;
	// for (int i = 0; i < nodes; ++i){
	//   set<int> s;
	//   s.insert(i);
	//   group.push_back(s);
	// }
	// groups.push_back(group);
	// groups.push_back(group);
	// groups.push_back(group);
	bool succ = loadAdiGroups(dir+"/QP/beam_ball_qp/0adi_groups.txt", groups);
	assert(succ);
  }

  typedef ADIPlanePreconSolver<double,FixedSparseMatrix<double> > Preconditioner;
  // typedef SingleADIPlanePreconSolver<double,FixedSparseMatrix<double> > Preconditioner;
  typedef MPRGPMonotonic<double, FixedSparseMatrix<double>, PlaneProjector<double>, Preconditioner > MPRGPSolver;

  const FixedSparseMatrix<double> SA(A);
  PlaneProjector<double> projector(planes_for_each_node, x);
  Preconditioner precond(SA, projector.getFace(), projector.getPlanes(), groups);
  MPRGPSolver solver(SA, B, precond, projector, max_it, tol);
  const int code = solver.solve(x);

  ERROR_LOG_COND("MPRGP is not convergent, result code is "<<code<<endl,code==0);
  DEBUG_FUN( MPRGPPlane<double>::checkResult(A, B, projector, x, tol) );
  assert( isFeasible(planes_for_each_node, x) );
}

void testQPFromFiles(const string qp_fold,const double tol,const int max_it,
					 const int T, const int delta_step=1, const bool no_con=false){

  assert_ge(delta_step,1);
  cout << "testQPFromFiles" << endl;

  {
	cout << "mprgp tol: " << tol << endl;
	cout << "mprgp max it: " << max_it << endl;
	cout << "init file: " << qp_fold << endl;
	cout << "total frames: "<< T << endl;
  }

  vector<double> desired_func_values;
  {
  	const string dir = "./test_case/data/";
  	const string fname = dir + qp_fold+"/0resulting_func_values.txt";
  	ifstream in(fname.c_str());
  	assert_ext( in.is_open(), fname );
  	int n = 0;
  	in >> n;
  	assert_in(T, 0, n);
  	desired_func_values.resize(n);
  	for (int i = 0; i < n; ++i){
  	  in >> desired_func_values[i];
  	  assert_eq(desired_func_values[i], desired_func_values[i]);
  	}
  	in.close();
  }

  for (int frame = 0; frame < T;frame += delta_step){

	cout << "step: " << frame << endl;
	ostringstream ossm_bin;
	ossm_bin << qp_fold + "/frame_" << frame << "_it_0.b";
	const double func_value = testQPFromFile( ossm_bin.str(), tol, max_it, no_con);
	cout << "diff: " << func_value - desired_func_values[frame] << endl;
	// assert_le(func_value, desired_func_values[frame]+1e-9);
	// assert_le(func_value, desired_func_values[frame]+1e-7);
  }

}

void testQPFromFiles(){

  // testQPFromFiles("/QP/dragon_plane_qp/", 1e-4, 1000, 150 ,20);
  // testQPFromFiles("/QP/dragon_stair_qp/", 1e-4, 1000, 150);
  // testQPFromFiles("/QP/dino_cylinder_qp/", 1e-4, 1000, 150);
  // testQPFromFiles("/QP/dino_ball_qp/", 1e-4, 1000, 150);
  // testQPFromFiles("/QP/beam_ball_qp/", 1e-4, 1000, 150);
}

void testLargeDragonQP(){

  cout << "testLargeDragonQP" << endl;

  const double tol = 1e-4;
  const int max_it = 10000;
  const bool no_con = false;
  {
	cout << "mprgp tol: " << tol << endl;
	cout << "mprgp max it: " << max_it << endl;
	cout << "init file: QP/dragon_large_plane_qp" << endl;
	cout << "total frames: "<< 7 << endl;
  }

  testQPFromFile( "/QP/dragon_large_plane_qp/frame_0_it_0.b", tol, max_it, no_con);
  testQPFromFile( "/QP/dragon_large_plane_qp/frame_10_it_0.b", tol, max_it, no_con);
  testQPFromFile( "/QP/dragon_large_plane_qp/frame_50_it_0.b", tol, max_it, no_con);
  testQPFromFile( "/QP/dragon_large_plane_qp/frame_93_it_0.b", tol, max_it, no_con);
  testQPFromFile( "/QP/dragon_large_plane_qp/frame_110_it_0.b", tol, max_it, no_con);
  testQPFromFile( "/QP/dragon_large_plane_qp/frame_260_it_0.b", tol, max_it, no_con);
  testQPFromFile( "/QP/dragon_large_plane_qp/frame_299_it_0.b", tol, max_it, no_con);
  
}

void comparePrecondConvergency(const string &file,const double tol, const int max_it, bool no_con = false){

  {
  	cout << "No Preconditioner\n";
  	typedef DiagonalPlanePreconSolver<double,FixedSparseMatrix<double>, true > Preconditioner;
  	testQPFromFile<Preconditioner>( file, tol, max_it, no_con);
  }

  {
  	cout << "Diagonal Preconditioner\n";
  	typedef DiagonalPlanePreconSolver<double,FixedSparseMatrix<double>,false > Preconditioner;
  	testQPFromFile<Preconditioner>( file, tol, max_it, no_con);
  }

  // {
  // 	cout << "Block diagonal Preconditioner\n";
  // 	typedef BlockDiagonalPlanePreconSolver<double,FixedSparseMatrix<double> > Preconditioner;
  // 	testQPFromFile<Preconditioner>( file, tol, max_it, no_con);
  // }

  {
  	cout << "Tridiagonal Preconditioner\n";
  	typedef TridiagonalPlanePreconSolver<double,FixedSparseMatrix<double> > Preconditioner;
  	testQPFromFile<Preconditioner>( file, tol, max_it, no_con);
  }

  // {
  // 	cout << "Symetric Gaussian-Seidel Preconditioner\n";
  // 	typedef SymGaussSeidelPlanePreconSolver<double,FixedSparseMatrix<double> > Preconditioner;
  // 	testQPFromFile<Preconditioner>( file, tol, max_it, no_con);
  // }

  {
	cout << "Cholesky Preconditioner\n";
	typedef CholeskyPlanePreconSolver<double,FixedSparseMatrix<double> > Preconditioner;
	testQPFromFile<Preconditioner>( file, tol, max_it, no_con);
  }

  {
  	cout << "ADI Preconditioner\n";
  	testADI( file, tol, max_it , no_con);
  }
}

void comparePrecondConvergency(){

  cout << "comparePrecondConvergency\n";
  FILE* f = NULL;
  // f = freopen("./tempt/dragon_asia_plane.txt","w,",stdout); assert(f);
  // comparePrecondConvergency("/QP/dragon_plane_qp/frame_78_it_0.b", 1e-2, 3000, true);

  // f = freopen("./tempt/dino_cyliner.txt","w,",stdout);  assert(f);
  // comparePrecondConvergency("/QP/dino_cylinder_qp/frame_100_it_0.b", 1e-4, 3000);

  // f = freopen("./tempt/dragon_asia_stair.txt","w,",stdout);  assert(f);
  // comparePrecondConvergency("/QP/dragon_stair_qp/frame_78_it_0.b", 1e-4, 3000);

  // f = freopen("./tempt/dino_ball.txt","w,",stdout);  assert(f);
  // comparePrecondConvergency("/QP/dino_ball_qp/frame_100_it_0.b", 1e-4, 3000);

  f = freopen("./tempt/beam_ball.txt","w,",stdout);  assert(f);
  comparePrecondConvergency("/QP/beam_ball_qp/frame_24_it_0.b", 1e-3, 3000, true);

  // f = freopen("./tempt/dragon_large.txt","w,",stdout);  assert(f);
  // comparePrecondConvergency("/QP/dragon_large_plane_qp/frame_93_it_0.b", 1e-4, 10000);
}

void testNoConQP(){

  cout << "testNoConQP" << endl;
  const double f1 = testQPFromFile( "/QP/dragon_plane_qp/frame_0_it_0.b", 1e-4, 196, true);
  assert_le(f1, -274.64494596);

  const double f2 = testQPFromFile( "/QP/dragon_plane_qp/frame_3_it_0.b", 1e-4, 256, true);
  assert_le(f2, -272.47107657);

  const double f3 = testQPFromFile( "/QP/dragon_plane_qp/frame_5_it_0.b", 1e-4, 311, true);
  assert_le(f3, -270.57578916);
}

#endif /* _TEST_SIMULATION_H_ */
