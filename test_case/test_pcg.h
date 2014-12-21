#ifndef _TEST_PCG_H_
#define _TEST_PCG_H_

#include <MPRGPSolver.h>
#include <FlexiblePCG.h>
using namespace MATH;

template<bool flexible = true>
void testpcgNoPrecond(const string &file_name,const double tol,const int max_it){

  const string dir = "./test_case/data/";
  
  SparseMatrix<double> A;
  VectorXd B, x;
  VVVec4d planes_for_each_node;
  const bool succ_QP = loadQP(A, B, planes_for_each_node, x, dir+file_name);
  const size_t n = planes_for_each_node.size();
  planes_for_each_node.clear();
  planes_for_each_node.resize(n);
  assert_ext(succ_QP, dir+file_name);

  typedef FlexiblePCG<double,SparseMatrix<double>,VectorXd,IdentityCGPrecond,flexible> PCG;
  IdentityCGPrecond precond;
  PCG solver(A, precond, tol, max_it);
  const int code = solver.solve(B, x);
  assert_eq(code ,0);
}

template<bool flexible = true>
void testpcgADI(const string &file_name,const double tol,const int max_it){

  const string dir = "./test_case/data/";
  
  SparseMatrix<double> A;
  VectorXd B, x;
  VVVec4d planes_for_each_node;
  vector<vector<set<int> > > groups;
  const bool succ_QP = loadQP(A, B, planes_for_each_node, x, dir+file_name);
  assert_ext(succ_QP, dir+file_name);
  const size_t n = planes_for_each_node.size();
  planes_for_each_node.clear();
  planes_for_each_node.resize(n);
  const bool succ_adi = loadAdiGroups(dir+"/QP/beam_ball_qp/0adi_groups.txt", groups);
  assert(succ_adi);

  typedef ADIPlanePreconSolver<double,FixedSparseMatrix<double> > Preconditioner;
  typedef FlexiblePCG<double, SparseMatrix<double>, VectorXd, Preconditioner, flexible> PCG;

  const FixedSparseMatrix<double> SA(A);
  PlaneProjector<double> projector(planes_for_each_node, x);
  Preconditioner precond(SA, projector.getFace(), projector.getPlanes(), groups);
  PCG solver(A, precond, tol, max_it);
  const int code = solver.solve(B, x);
  assert_eq(code ,0);
}

template<bool flexible = true>
void testpcgDiag(const string &file_name,const double tol,const int max_it){

  const string dir = "./test_case/data/";
  
  SparseMatrix<double> A;
  VectorXd B, x;
  VVVec4d planes_for_each_node;
  vector<vector<set<int> > > groups;
  const bool succ_QP = loadQP(A, B, planes_for_each_node, x, dir+file_name);
  assert_ext(succ_QP, dir+file_name);
  const size_t n = planes_for_each_node.size();
  planes_for_each_node.clear();
  planes_for_each_node.resize(n);
  const bool succ_adi = loadAdiGroups(dir+"/QP/beam_ball_qp/0adi_groups.txt", groups);
  assert(succ_adi);

  typedef DiagonalPlanePreconSolver<double,FixedSparseMatrix<double>,false > Preconditioner;
  typedef FlexiblePCG<double, SparseMatrix<double>, VectorXd, Preconditioner, flexible> PCG;

  const FixedSparseMatrix<double> SA(A);
  PlaneProjector<double> projector(planes_for_each_node, x);
  Preconditioner precond(SA, projector.getFace(), projector.getPlanes());
  PCG solver(A, precond, tol, max_it);
  const int code = solver.solve(B, x);
  assert_eq(code ,0);
}

void testpcg(){

  FILE *f = NULL;
  f = freopen("./tempt/beam_ball_pcg.txt","w,",stdout);  assert(f);

  testpcgNoPrecond("/QP/beam_ball_qp/frame_24_it_0.b", 1e-3, 3000);
  cout << "monotonic mprgp solving:\n";

  testpcgNoPrecond<false>("/QP/beam_ball_qp/frame_24_it_0.b", 1e-3, 3000);
  cout << "monotonic mprgp solving:\n";

  testpcgDiag("/QP/beam_ball_qp/frame_24_it_0.b", 1e-3, 3000);
  cout << "monotonic mprgp solving:\n";

  testpcgDiag<false>("/QP/beam_ball_qp/frame_24_it_0.b", 1e-3, 3000);
  cout << "monotonic mprgp solving:\n";

  testpcgADI("/QP/beam_ball_qp/frame_24_it_0.b", 1e-3, 3000);
  cout << "monotonic mprgp solving:\n";

  testpcgADI<false>("/QP/beam_ball_qp/frame_24_it_0.b", 1e-3, 3000);
  cout << "monotonic mprgp solving:\n";

}

#endif /* _TEST_PCG_H_ */
