#ifndef _ACTIVESETQP3D_H_
#define _ACTIVESETQP3D_H_

#include <vector>
#include <eigen3/Eigen/Dense>
#include <MPRGPUtility.h>
#include <iomanip>
using namespace Eigen;
using namespace std;

namespace MATH{

  typedef Eigen::Vector4d Vec4d;
  typedef Eigen::Vector3d Vec3d;
  typedef Eigen::Vector3i Vec3i;
  typedef Eigen::Matrix3d Mat3d;
  typedef Eigen::Matrix2d Mat2d;
  typedef vector<Vector4d,aligned_allocator<Vector4d> > VVec4d;

  // the plane is defined as p[0:2].dot(y)+p[3]=0.
  inline double dist(const Vec4d& p,const Vec3d& v){
	assert_eq(v,v);
	return v.dot(p.block<3,1>(0,0))+p[3];
  }

  inline bool isFeasible(const VVec4d& p,const Vec3d& v){

	assert_eq(v,v);
	size_t nrP=(size_t)p.size();
	for(size_t i=0;i<nrP;i++)
	  if(dist(p[i],v) < -ScalarUtil<double>::scalar_eps){
		DEBUG_LOG("dist(p,v): " << dist(p[i],v));
		return false;
	  }
	return true;
  }

  inline bool isFeasible(const VVec4d& p,const VectorXd& v){

	assert_eq(v,v);
	assert_eq(v.size()%3,0);
	for (int i = 0; i < v.size(); i+=3){
	  const Vec3d vi = v.segment(i,3);
	  if (!isFeasible(p,vi)){
		DEBUG_LOG("point "<<i/3<<" is infeasible: " << vi.transpose());
		return false;
	  }
	}
	return true;
  }

  inline bool findFeasible(const VVec4d& p,Vec3d& v){

	v.setZero();
	assert_eq(v,v);
	size_t nrP=(size_t)p.size();
	std::vector<double> weight(nrP,1.0f);
	for(size_t iter=0;iter<100*nrP;iter++){

	  Mat3d H=Mat3d::Zero();
	  Vec3d G=Vec3d::Zero();
	  for(size_t i=0;i<nrP;i++){

		assert_eq(v,v);
		double E=weight[i]*std::exp(-dist(p[i],v));
		H+=p[i].block<3,1>(0,0)*p[i].block<3,1>(0,0).transpose()*E;
		G-=p[i].block<3,1>(0,0)*E;
	  }
	  if(std::abs(H.determinant()) < ScalarUtil<double>::scalar_eps)
		H.diagonal().array() += ScalarUtil<double>::scalar_eps;
	  v-=H.inverse()*G;

	  double minDist=0.0f;
	  size_t minId=-1;
	  for(size_t i=0;i<nrP;i++){
		assert_eq(v,v);
		double currDist=dist(p[i],v);
		if(currDist < minDist){
		  minDist=currDist;
		  minId=i;
		}
	  }
	  if(minId == -1)break;
	  weight[minId]*=2.0f;
	}
	assert_eq(v,v);
	return isFeasible(p,v);
  }

  //solving the distance QP problem in 3D, the formulation is:
  //	min_{curr}	\|v-v0\|^2
  //	s.t.		\forall i, v is in front of plane p_i
  //
  //you should provide:
  //an initial guess:	v
  //the active set:	aSet
  //
  //here the initial guess must be feasible or you should call:
  //findFeasible(p,v)
  inline bool findClosestPoint(const VVec4d& p,const Vec3d& v0,Vec3d& v,Vec3i& aSet,double eps=1E-18){

	assert(isFeasible(p,v));
	assert_eq(v,v);
	assert_eq(v0,v0);

	//rearrange
	char nrA=0;
	int nrP=(int)p.size();
	vector<bool> aTag(nrP,false);
	for(char d=0;d<3;d++){
	  if(aSet[d] != -1){
		assert_in(aSet[d],0,aTag.size()-1);
		aTag[aSet[d]]=true;
		aSet[nrA++]=aSet[d];
	  }
	}

	//forward decl for mainIter
	int minA = -1;
	double minLambda,alphaK,nDotDir,distP;

	Mat2d M2;
	Mat3d A,M3;
	Vec3d dir,lambda;
	dir.setZero();
	lambda.setZero();

	const int max_it = 1000;
	for (int it = 0; it < max_it; ++it){

	  //step 1: solve the following equation:
	  // I*v + A^T*\lambda = v0
	  // A*v + d = 0
	  //where we use schur complementary method
	  
	  for(char d=0;d<nrA;d++){
		lambda[d]=dist(p[aSet[d]],v0);
	  }

	  if(nrA == 0){
		dir=v0;
		dir-=v;
	  }else if(nrA == 1){
		//the plane's normal has already been normalized
		dir=v0-p[aSet[0]].block<3,1>(0,0)*lambda[0];
		dir-=v;
	  }else if(nrA == 2){
		A.row(0)=p[aSet[0]].block<3,1>(0,0);
		A.row(1)=p[aSet[1]].block<3,1>(0,0);
		M2=A.block<2,3>(0,0)*A.block<2,3>(0,0).transpose();
		lambda.block<2,1>(0,0)=M2.llt().solve(lambda.block<2,1>(0,0));
		dir=v0-A.block<2,3>(0,0).transpose()*lambda.block<2,1>(0,0);
		dir-=v;
	  }else if(nrA == 3){
		A.row(0)=p[aSet[0]].block<3,1>(0,0);
		A.row(1)=p[aSet[1]].block<3,1>(0,0);
		A.row(2)=p[aSet[2]].block<3,1>(0,0);
		M3=A*A.transpose();
		lambda=M3.llt().solve(lambda);
		dir.setZero();	//in that case, no dir can be allowed
	  }

	  //step 2: test stop if p is very small
	  if(dir.squaredNorm() < eps){
		minA=-1;
		minLambda=0.0f;
		for(char d=0;d<nrA;d++)
		  if(lambda[d] > minLambda){
			minA=d;
			minLambda=lambda[d];
		  }
		//aha, we have all negative lagrangian multiplier, exit now!
		if(minA == -1){
		  for (int k = nrA; k < 3; ++k)
			aSet[k] = -1;
		  return true;
		}
		//for the most positive component, we remove it from active set
		if(nrA > 1){
		  aTag[minA]=false;
		  aSet[minA]=aSet[nrA-1];
		}
		aSet[nrA-1]=-1;
		nrA--;
	  }else{
		//step 3: move until we are blocked
		assert_le(nrA,2);
		minA=-1;
		alphaK=1.0f;
		for(int i=0;i<nrP;i++){
		  if (aTag[i])
			continue;
		  nDotDir=p[i].block<3,1>(0,0).dot(dir);
		  if(nDotDir < 0.0f){
			assert_eq(v,v);
			distP=dist(p[i],v);
			if(distP <= 0.0f){
			  alphaK=0.0f;
			  minA=i;
			  break;
			}else{
			  distP/=-nDotDir;
			  if(distP < alphaK){
				alphaK=distP;
				minA=i;
			  }
			}
		  }
		}
		v+=alphaK*dir;

		if(minA >= 0){
		  //already in active set, so this is rounding error
		  for(char d=0;d<nrA;d++)
			if(minA == aSet[d]){
			  cout << "error: " << "rounding error\n";
			  cout << setprecision(14) << "v0: " << v0.transpose() << endl;
			  cout << setprecision(14) << "v: " << v.transpose() << endl;
			  cout << "s: " << aSet.transpose() << endl;
			  const Vec3d n = p[minA].head(3);
			  const double b = p[minA][3];
			  cout << "dv0: "<< n.dot(v0)+b << endl;
			  cout << "dv: "<< n.dot(v)+b << endl;
			  return false;
			}
		  //expand active set
		  aTag[minA]=true;
		  aSet[nrA++]=minA;
		}
	  }
	}

	cout<<"error: not convergent for "<<max_it<<" iterations in finding closest point."<<endl;
	cout<< setprecision(14) << "v0: " << v0.transpose() << endl;
	cout<< setprecision(14) << "v: " << v.transpose() << endl;
	cout<< "s: " << aSet.transpose() << endl;
	cout<< "minA: " << minA << endl;
	if (minA >= 0){
	  const Vec3d n = p[minA].head(3);
	  const double b = p[minA][3];
	  cout << "dv0: "<< n.dot(v0)+b << endl;
	  cout << "dv: "<< n.dot(v)+b << endl;
	}

	for (int k = nrA; k < 3; ++k)
	  aSet[k] = -1;
	return false;
  }


  // The active set method for helping to compute BETA, which solves:
  // 
  // \beta = min 1/2*||(-beta)-(-g)||_2^2 
  //               s.t. 
  //    (-beta)*n[j] >=0 for j in f, 
  //               and 
  //           beta*phi=0.
  // 
  // The special case where phi=0 should be taken carefully.
  inline bool findClosestPoint(const VVec4d& p,const vector<int>&f,const Vec3d&g,const Vec3d& phi,Vec3d& beta,double eps=1E-18){

	bool succ = false;
	assert_ge(f.size(),2);

	Vec3i aSet;
	aSet.setConstant(-1);
	vector<Vector4d,aligned_allocator<Vector4d> > planes;
	planes.reserve(f.size());
	for (int i = 0; i < f.size(); ++i){
	  assert_in(f[i],0,p.size()-1);
	  planes.push_back(p[f[i]]);
	  planes[i].head(3) *= -1.0f;
	  planes[i][3]=0; 
	}

	Vec3d v0 = g;
	const double pnorm = phi.norm();
	if ( 2 == f.size() && pnorm >= ScalarUtil<double>::scalar_eps ){
	  // project v0on to the plane defined by phi.
	  v0 -= v0.dot(phi)*phi*(1.0f/(pnorm*pnorm));
	}

	const bool found = findFeasible(planes,beta);
	if(!found){
	  ERROR_LOG("can not found a feasible point. return: "<< beta.transpose());
	}

	succ = findClosestPoint(planes,v0,beta,aSet,eps);
	return succ;
  }

}// end of namespace

#endif /* _ACTIVESETQP3D_H_ */
