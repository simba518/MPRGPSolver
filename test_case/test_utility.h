#ifndef _TEST_UTILITY_H_
#define _TEST_UTILITY_H_

#include "test_solver.h"
using namespace MATH;

typedef vector<Eigen::Matrix<double,4,1>, Eigen::aligned_allocator<Eigen::Matrix<double,4,1> > > VVec4Xd;

void test_io(){
  
  const string file_name = "./tempt_test_io.mat";

  const int n = 10;
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

#endif /* _TEST_UTILITY_H_ */
