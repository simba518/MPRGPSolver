#include "ActiveSetQP3D.h"
using namespace MATH;

void test_findClosestPoint(){

  cout << "test_findClosestPoint()\n";
  
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
  assert_le(sqrt(n.dot(v)+b), 1e-8);

  assert(found);

}

void test_findClosestPoint2(){

  cout << "test_findClosestPoint2()\n";
  
  vector<Vector4d,aligned_allocator<Vector4d> > planes;
  Vector4d p;
  p << 1,1,0,0.4*sqrt(2.0f)/2.0f;
  p.head(3) /= p.head(3).norm();
  planes.push_back(p);

  Vec3i aSet, c_aSet;
  c_aSet << 0,-1,-1;
  aSet.setConstant(-1);

  Vec3d v0, v;
  v0 <<  -0.043433492549893,  -0.35835273831612, -0.058816671017984;
  v << 0.080596472022753, -0.21225671260567,  0.12222764966631;

  const bool found = findClosestPoint(planes,v0,v,aSet);
  assert_eq(c_aSet, aSet);
  
  const Vec3d n = planes[0].head(3);
  const double b = planes[0][3];
  assert_le(sqrt(n.dot(v)+b), 1e-8);

  assert(found);

}

void test_findFeasiblePoint(){

  cout << "test_findFeasiblePoint()\n";
  {
	Vec3d v;
	v << 3.07430376886, 7.82701969387, -2.38041048088;
	VVec4d p(2);
	p[0] << 0.18497331977, 0.95654901619, -0.225386003556, 0;
	p[1] << -0.804077659485, -0.521780805583,  0.284962994863, 0;
	assert( findFeasible(p, v, true) );
  }

}
