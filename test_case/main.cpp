#include "test_solver.h"
#include "test_projection.h"
#include "test_ActiveSet3D.h"

#include <float.h>
float ScalarUtil<float>::scalar_max=FLT_MAX;
float ScalarUtil<float>::scalar_eps=1E-5f;
double ScalarUtil<double>::scalar_max=DBL_MAX;
double ScalarUtil<double>::scalar_eps=1E-9;

int main(int argc, char *argv[]){

  test_LowerBoundProjector();
  test1DSolverLB();
  test2DSolverLB();
  test1DSolverBB();
  test2DSolverBB();
  return 0;
}
