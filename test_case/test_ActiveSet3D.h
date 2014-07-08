#include "ActiveSetQP3D.h"
using namespace MATH;

void test_findClosestPoint(){
  
  vector<Vector4d,aligned_allocator<Vector4d> > planes;
  Vector4d p;
  p << 1,1,0,1;
  p.head(3) = p.head(3)/p.head(3).norm();
  planes.push_back(p);

  Vec3i aSet, c_aSet;
  c_aSet << 0,-1,-1;
  aSet.setConstant(-1);
  Vec3d v0, v;
  v0 << -1.37611848757945,   -0.03815673149568, -0.0897734617869561;
  v.setZero();
  const bool found = findClosestPoint(planes,v0,v,aSet);
  assert_eq(c_aSet, aSet);
  
  const Vec3d n = planes[0].head(3);
  const double b = planes[0][3];
  assert_le(sqrt(n.dot(v)+b), 1e-12);

  assert(found);

}
