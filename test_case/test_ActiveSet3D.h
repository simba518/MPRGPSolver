#include "ActiveSetQP3D.h"
using namespace MATH;

void test_findClosestPoint(){
  
  vector<Vector4d,aligned_allocator<Vector4d> > planes;
  Vector4d p;
  p << 1,1,0,1;
  p.head(3) = p.head(3)/p.head(3).norm();
  planes.push_back(p);

  Vec3i aSet;
  aSet.setConstant(-1);

  Vec3d v0, v;
  v0 << -1.37611848757945,   -0.03815673149568, -0.0897734617869561;
  v.setZero();

  const bool found = findClosestPoint(planes,v0,v,aSet);

  cout<< "v0: " << v0.transpose() << endl;
  cout<< "v: " << v.transpose() << endl;
  cout<< "s: " << aSet.transpose() << endl;

  const Vec3d n = planes[0].head(3);
  const double b = planes[0][3];
  cout << "dv0: "<< n.dot(v0)+b << endl;
  cout << "dv: "<< n.dot(v)+b << endl;

  assert(found);

}
