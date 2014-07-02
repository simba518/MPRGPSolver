#ifndef _TEST_PROJECTION_H_
#define _TEST_PROJECTION_H_

#include <MPRGPProjection.h>
using namespace MATH;

void test_LowerBoundProjector(){

  cout << "test projector for x >= L\n";
  
  // init
  VectorXd L(2);
  L << 1,2;
  LowerBoundProjector<double> P(L);
  assert_eq(P.getFace().size(),2);

  // step limit
  VectorXd D(2), X(2);
  D << 2,2;
  D *= 0.5f;
  X << 2,4;
  assert_eq(P.stepLimit(X, D),1.0f);

  // project
  VectorXd Y(2);
  P.project(X,Y);
  assert_eq(Y,X);
  X.setZero();
  P.project(X,Y);
  assert_eq(Y,L);
  
  // decide face
  X << 1,3;
  P.DECIDE_FACE(X);
  assert_eq(P.getFace()[0],-1);
  assert_eq(P.getFace()[1],0);

  // PHI
  VectorXd g(2), phi(2);
  g << -2,3;
  P.PHI(g,phi);
  assert_eq(phi[0],0);
  assert_eq(phi[1],3);

  // BETA
  VectorXd beta(2);
  P.BETA(g,beta);
  assert_eq(beta[0],-2);
  assert_eq(beta[1],0);

  g << 2,3;
  P.BETA(g,beta);
  assert_eq(beta[0],0);
  assert_eq(beta[1],0);

}

#endif /* _TEST_PROJECTION_H_ */

