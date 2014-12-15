#ifndef _TEST_SIMULATION_H_
#define _TEST_SIMULATION_H_

#include <MPRGPSolver.h>
#include "test_utility.h"
using namespace MATH;

double testQPFromFile(const string &file_name,const double tol, const int max_it){

  const string dir = "./test_case/data/";
  
  SparseMatrix<double> A;
  VectorXd B, x;
  VVVec4d planes_for_each_node;
  const bool succ_to_load_QP = loadQP(A, B, planes_for_each_node, x, dir+file_name);
  assert_ext(succ_to_load_QP, dir+file_name);
  cout << "dimension: " << B.size() << endl;

  PlaneProjector<double> projector(planes_for_each_node, x);
  const FixedSparseMatrix<double> SA(A);
  const int code = MPRGPPlane<double>::solve(SA,B,projector,x,tol,max_it);
  ERROR_LOG_COND("MPRGP is not convergent, result code is "<<code<<endl,code==0);
  DEBUG_FUN( MPRGPPlane<double>::checkResult(A, B, projector, x, tol) );

  const double fun = (x.dot(A*x))*0.5f-x.dot(B);
  cout<< setprecision(12) << "function value: " << fun << endl;
  assert( isFeasible(planes_for_each_node, x) );
  return fun;
}

void testQPFromFiles(){

  cout << "testQPFromFiles" << endl;
  
  const string qp_fold = "/dragon_asia_qp/";
  const double tol = 1e-4;
  const int max_it = 1000;
  const int T = 20;

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

  for (int frame = 0; frame < T; ++frame){

	cout << "step: " << frame << endl;
	ostringstream ossm_bin;
	ossm_bin << qp_fold + "/frame_" << frame << "_it_0.b";
	const double func_value = testQPFromFile( ossm_bin.str(), tol, max_it);
	assert_le(func_value, desired_func_values[frame]+1e-9);
  }

}

void testFuncValue(){

  cout << "testFuncValue" << endl;
  const double f1 = testQPFromFile( "/dragon_asia_qp/frame_0_it_0.b", 1e-4, 1000);
  assert_le(f1, -274.64494596);

  const double f2 = testQPFromFile( "/dragon_asia_qp/frame_3_it_0.b", 1e-4, 1000);
  assert_le(f2, -272.47066022);

  const double f3 = testQPFromFile( "/dragon_asia_qp/frame_5_it_0.b", 1e-4, 1000);
  assert_le(f3, -270.5750235);
}

#endif /* _TEST_SIMULATION_H_ */
