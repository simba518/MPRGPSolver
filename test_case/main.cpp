// #include "test_solver.h"
// #include "test_projection.h"
#include "test_ActiveSet3D.h"
// #include "test_utility.h"
// #include "test_simulation.h"
// #include "test_precondition.h"
// #include "test_pcg.h"
using namespace MATH;

#include <float.h>
float ScalarUtil<float>::scalar_max=FLT_MAX;
float ScalarUtil<float>::scalar_eps=1E-5f;
double ScalarUtil<double>::scalar_max=DBL_MAX;
double ScalarUtil<double>::scalar_eps=1E-9;

int main(int argc, char *argv[]){

  // test_io();
  test_findClosestPoint();
  test_findClosestPoint2();
  test_findFeasiblePoint();
  // test_LowerBoundProjector();
  // test_PlaneProjector();
  // test_OnePlaneProjector();

  // test1DSolverLB();
  // test2DSolverLB();
  // test1DSolverBB();
  // test2DSolverBB();
  // testMPRGPPlaneSolver3D();
  // testMPRGPPlaneSolver3D_OnePlane();
  // testSolverFromFile();
  // testComputeLagMultipliers();

  // test_tridiagonalPrecond();
  // testNoConQP();

  // testQPFromFiles();
  // testLargeDragonQP();
  
  // comparePrecondConvergency();
  // testpcg();

  return 0;
}
