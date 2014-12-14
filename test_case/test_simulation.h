#ifndef _TEST_SIMULATION_H_
#define _TEST_SIMULATION_H_

#include <MPRGPSolver.h>
#include "test_utility.h"
using namespace MATH;

void testQPFromFile(const string &file_name,const double tol, const int max_it){

  const string dir = "./test_case/data/";
  
  SparseMatrix<double> A;
  VectorXd B, x;
  VVVec4d planes_for_each_node;
  const bool succ_to_load_QP = loadQP(A, B, planes_for_each_node, x, dir+file_name);
  assert_ext(succ_to_load_QP, dir+file_name);
  cout << "dimension: " << B.size() << endl;

  PlaneProjector<double> projector(planes_for_each_node, x);
  FixedSparseMatrix<double> SA(A);
  cout<< "in: " << x.norm() << endl;
  const int code = MPRGPPlane<double>::solve(SA,B,planes_for_each_node,x,tol,max_it);
  cout<< "out: " << x.norm() << endl;

  ERROR_LOG_COND("MPRGP is not convergent, result code is "<<code<<endl,code==0);
  DEBUG_FUN( MPRGPPlane<double>::checkResult(A, B, projector, x, tol) );

}

void testQPFromFiles(){

  cout << "testQPFromFiles" << endl;
  
  const string qp_fold = "/dragon_asia_qp/";
  const double tol = 1e-4;
  const int max_it = 1000;
  const int T = 150;

  {
	cout << "mprgp tol: " << tol << endl;
	cout << "mprgp max it: " << max_it << endl;
	cout << "init file: " << qp_fold << endl;
	cout << "total frames: "<< T << endl;
  }

  for (int frame = 0; frame < 150; ++frame){

	cout << "step: " << frame << endl;
	ostringstream ossm_bin;
	ossm_bin << qp_fold + "/frame_" << frame << "_it_0.b";
	testQPFromFile( ossm_bin.str(), tol, max_it);
  }

}

#endif /* _TEST_SIMULATION_H_ */
