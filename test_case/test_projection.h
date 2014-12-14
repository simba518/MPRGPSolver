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
  assert_eq((int)P.getFace()[0],2);
  assert_eq((int)P.getFace()[1],0);

  // PHI
  VectorXd g(2), phi(2);
  g << -2,3;
  P.PHI(g,phi);
  assert_eq(phi[0],0);
  assert_eq(phi[1],3);

  // BETA
  VectorXd beta(2);
  P.BETA(g,beta,phi);
  assert_eq(beta[0],-2);
  assert_eq(beta[1],0);

  g << 2,3;
  P.BETA(g,beta,phi);
  assert_eq(beta[0],0);
  assert_eq(beta[1],0);

}

void test_PlaneProjector(){
  
  cout << "test projector for xi*nj >= bj\n";
  typedef Eigen::Matrix<double,4,1> Vec4X;
  typedef vector<Vec4X,Eigen::aligned_allocator<Vec4X> > VVec4X;
  typedef vector<VVec4X > VVVec4X;

  // initialize
  VectorXd x(2*3), y(2*3);
  VVec4X planes;
  VVVec4X planes_for_each_node;
  {
	
	Vector4d p;
	p << 1,0,0,0;
	p.head(3) /= p.head(3).norm();
	planes.push_back(p);

	p << 0,1,0,0;
	p.head(3) /= p.head(3).norm();
	planes.push_back(p);

	p << 0,0,1,0;
	p.head(3) /= p.head(3).norm();
	planes.push_back(p);

	p << -1,-1,-1,10;
	p.head(3) /= p.head(3).norm();
	planes.push_back(p);
  }

  convert<double>(planes, planes_for_each_node, x.size()/3);
  assert(findFeasible(planes_for_each_node, x));
  PlaneProjector<double> P(planes_for_each_node, x);
  assert_eq((int)P.getFace().size(),(int)x.size());
  for (int i = 0; i < (int)P.getFace().size(); ++i)
    assert_eq(P.getFace()[i],0);

  // test step limit
  {
	VectorXd d(x.size());
	d << 0.2,0.2,0.2,
	  0.1,0.0,0.0;

	x << 0.1,0.1,0.1,
	  0.2,0.2,0.2;
	const double t = P.stepLimit(x,d);
	assert_eq(t,0.5);
  }

  // test project
  {
	// feasible points
  	VectorXd px;
	x << 0.1,0.1,0.1,
	  0.2,0.2,0.2;
  	P.project(x, px);
  	assert_le((px-x).norm(),1e-12);

	// infeasible points
  	x << -0.1,-0.1,-0.1,
  	  -0.2,0.2,0.2;
  	P.project(x, px);
  	y << 0,0,0,  0,0.2,0.2;
  	assert_le((px-y).norm(),1e-12);

	// infeasible points, project the line of two planes.
  	x << -0.1,-0.1,0.1,
  	  0.2,-0.2,-0.2;
  	P.project(x, px);
  	y << 0,0,0.1,  0.2,0,0;
  	assert_le((px-y).norm(),1e-12);
  }

  // test decide face
  {
	y << -0.0,-0.0,-0.0,  -0.0,0.2,0.2;
	P.DECIDE_FACE(y);
	const vector<char> &face = P.getFace();
	assert_eq(face[0],3);
	assert_eq(face[1],3);
	assert_eq(face[2],3);
	assert_eq(face[3],1);
	assert_eq(face[4],1);
	assert_eq(face[5],1);

	// y[0] is on 2 plane,
	// y[1] is on 1 plane.
	y << -0.0,0.1,-0.0,  -0.0,0.2,0.2;
	P.DECIDE_FACE(y);
	const vector<char> &face2 = P.getFace();
	assert_eq(face2[0],2);
	assert_eq(face2[1],2);
	assert_eq(face2[2],2);
	assert_eq(face2[3],1);
	assert_eq(face2[4],1);
	assert_eq(face2[5],1);
  }

  // test PHI, BETA
  {
  	VectorXd g(6),phi(6),beta(6);

  	// y[0] is on 1 plane,
  	// y[1] is free.
  	y << -0.0,0.1,0.1,  0.1,0.1,0.2;
	g << -0.2,-0.2,-0.2,  -0.1,-0.1,-0.1;
  	P.DECIDE_FACE(y);
  	P.PHI(g, phi);
  	P.BETA(g,beta,phi);

  	x << 0,-0.2,-0.2,  -0.1,-0.1,-0.1;
  	assert_le((x-phi).norm(), 1e-12);
  	x << -0.2,0,0,   0,0,0;
  	assert_le((x-beta).norm(), 1e-12);

  	// y[0] is on 3 planes,
  	// y[1] is free.
  	y << -0.0,-0.0,-0.0,  0.1,0.1,0.2;
	g << -0.2,0.2,0.2,  -0.1,-0.1,-0.1;
  	P.DECIDE_FACE(y);
  	P.PHI(g, phi);
  	P.BETA(g,beta,phi);

  	x << 0,0,0, -0.1,-0.1, -0.1;
  	assert_le((x-phi).norm(), 1e-12);
  	x << -0.2,0,0,   0,0,0;
  	assert_le((x-beta).norm(), 1e-12);

  	// test phi == 0
	y << -0.0,-0.0,1.0,  0.1,0.1,0.2;
  	g << -0.2,-0.2,-0.2,  -0.1,-0.1,-0.1;
  	P.DECIDE_FACE(y);
  	phi.setZero();
  	P.BETA(g,beta,phi);
  	x << -0.2,-0.2,-0.2,  0,0,0;
  	assert_le((x-beta).norm(), 1e-12);

  	// y[0] is on 2 planes,
  	// y[1] is on 2 planes, but phi[1] = 0.
  	y << 0.1,-0.0,-0.0,  -0.0,-0.0,0.2;
  	g << -0.2,-0.2,-0.2,  -0.1,-0.1,-0.1;
  	P.DECIDE_FACE(y);

  	P.PHI(g, phi);
  	x << -0.2,0,0,   0,0,-0.1;
  	assert_le((x-phi).norm(), 1e-12);
  	phi.tail(3).setZero();

  	P.BETA(g,beta,phi);
  	x << 0.0f,-0.2,-0.2,  -0.1,-0.1,-0.1;
  	assert_le((x-beta).norm(), 1e-12);
  }

}

void test_OnePlaneProjector(){
  
  cout << "test projector for xi*nj >= bj\n";

  // initialize
  typedef Eigen::Matrix<double,4,1> Vec4X;
  typedef vector<Vec4X,Eigen::aligned_allocator<Vec4X> > VVec4X;
  typedef vector<VVec4X > VVVec4X;

  // initialize
  VectorXd x(3);
  VVec4X planes;
  VVVec4X planes_for_each_node;
  {
	Vector4d p;
	p << 1,1,0,sqrt(2.0)*0.5f;
	p.head(3) /= p.head(3).norm();
	planes.push_back(p);
  }

  
  convert<double>(planes, planes_for_each_node, x.size()/3);
  assert(findFeasible(planes_for_each_node, x));
  PlaneProjector<double> P(planes_for_each_node, x);
  assert_eq((int)P.getFace().size(),(int)x.size());
  for (int i = 0; i < (int)P.getFace().size(); ++i)
    assert_eq(P.getFace()[i],0);

  // test step limit
  {
	x << 0,2,0;
	VectorXd d(x.size());
	d << 1,0.0,0.0;
	const double t = P.stepLimit(x,d);
	assert_le(abs(t-3.0),1e-12);
  }

  // test PHI, BETA, infeasible point.
  {
  	x << 0,-1.0,0;
  	P.DECIDE_FACE(x);
  	VectorXd g(3),phi(3),beta(3),c(3);
  	g << 0,-1,0;
  	P.PHI(g, phi);
  	P.BETA(g,beta,phi);
  	c << 0.5,-0.5,0;
  	assert_le( (phi-c).norm(), 1e-12  );
  	c << -0.5,-0.5,0;
  	assert_le( (beta-c).norm(), 1e-12  );
  }

  // test PHI, BETA, feasible point.
  {
  	x << 0,-0.9,0;
  	P.DECIDE_FACE(x);
  	VectorXd g(3),phi(3),beta(3),c(3);
  	g << 0,-1,0;
  	P.PHI(g, phi);
  	P.BETA(g,beta,phi);
  	c << 0,-1,0;
  	assert_le( (phi-c).norm(), 1e-12  );
  	c << 0,0,0;
  	assert_le( (beta-c).norm(), 1e-12  );
  }

  // test projection
  {
  	x << 0,-2.0,0;
  	VectorXd y(3);
  	P.project(x,y);
  	VectorXd c_y(3);
  	c_y << 0.5, -1.5, 0.0f;
  	assert_le((y-c_y).norm(),1e-12);
  }
}

#endif /* _TEST_PROJECTION_H_ */

