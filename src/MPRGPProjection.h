#ifndef _MPRGPPROJECTION_H_
#define _MPRGPPROJECTION_H_

#include <stdio.h>
#include <vector>
#include <eigen3/Eigen/Sparse>
#include <eigen3/Eigen/Dense>
#include <MPRGPUtility.h>
#include <ActiveSetQP3D.h>
using namespace Eigen;
using namespace std;

namespace MATH{

  // PROJECTOIN ----------------------------------------------------------------
  // support only lower bound constraints such that: x >= L.
  template <typename T>
  class LowerBoundProjector{

	typedef Eigen::Matrix<T,-1,1> Vec;
	
  public:
    LowerBoundProjector(const Vec &L):_L(L){
	  _face.resize(_L.size());
	  _face.assign(_L.size(),0);
	}
	const vector<char> &getFace()const{
	  return _face;
	}

	// return the largest step in direction -D.
	T stepLimit(const Vec& X,const Vec& D, const T alpha_cg=ScalarUtil<T>::scalar_max) const{

	  assert_eq(D.size(), X.size());
	  assert_eq(_L.size(), X.size());
	  T ret=ScalarUtil<T>::scalar_max;
	  T tmp;
#pragma omp parallel private(tmp)
	  {
		tmp=ScalarUtil<T>::scalar_max;
#pragma omp for
		for(size_t i=0;i<X.size();i++){
		  if(D[i] > ScalarUtil<T>::scalar_eps && X[i] > _L[i])	//handle rounding err
			tmp=std::min<T>(tmp,(X[i]-_L[i])/D[i]);
		}

		OMP_CRITICAL_
		  ret=std::min<T>(ret,tmp);
	  }
	  return ret;
	}

	// project the point 'in' onto the feasible domain.
	void project(const Vec& in,Vec& out) const{
	  assert_eq(in.size(), _L.size());
	  out.resize(_L.size());
	  OMP_PARALLEL_FOR_
		for(size_t i=0;i<in.size();i++)
		  out[i]=std::max<T>(in[i],_L[i]);
	}

	void PHI(const Vec& in,Vec& out){
	  MASK_FACE(in,out,_face);
	}

	void BETA(const Vec& in,Vec& out, const Vec&phi){

	  assert_eq(in.size(), _face.size());
	  out.resize(in.size());
	  OMP_PARALLEL_FOR_
		for(size_t i=0;i<in.rows();i++){
		  if( 0 == _face[i])
			out[i]=0.0f;
		  else
			out[i]=std::min<T>(in[i],0.0f);
		}
	}

	void DECIDE_FACE(const Vec& x){

	  assert_eq(x.size(), _L.size());
	  assert_eq(x.size(), _face.size());
	  const Vec& L = _L;
	  _face.assign(x.size(),0);
	  OMP_PARALLEL_FOR_
		for(size_t i=0;i<x.size();i++){
		  if(abs(x[i]-L[i]) < ScalarUtil<T>::scalar_eps)
			_face[i]=2;
		}

	}

	T PHITPHI(const Vec& x,const T&alphaBar,const Vec&phi){

	  assert_eq(x.size(), _L.size());
	  assert_eq(x.size(), phi.size());
	  const Vec &L = _L;
	  T phiTphi=0.0f;
#pragma omp parallel for reduction(+:phiTphi)
	  for(size_t i=0;i<x.rows();i++){
		T phiTilde=0.0f;
		if(phi[i] > 0.0f && x[i] > L[i])	//handle rounding error
		  phiTilde=std::min<T>((x[i]-L[i])/alphaBar,phi[i]);
		assert_ge(phiTilde*phi[i], 0.0f);
		phiTphi+=phiTilde*phi[i];
	  }
	  return phiTphi;
	}
	
  private:
	const Vec &_L;
	vector<char> _face;
  };

  // support box boundary constraints such that: H >= x >= L.
  template <typename T>
  class BoxBoundProjector{

	typedef Eigen::Matrix<T,-1,1> Vec;
	
  public:
	BoxBoundProjector(const Vec &L, const Vec &H):_L(L),_H(H){
	  assert_eq(L.size(), H.size());
	  _face.resize(_L.size());
	  _face.assign(_L.size(), 0);
	}
	const vector<char> &getFace()const{
	  return _face;
	}

	T stepLimit(const Vec& X,const Vec& D, const T alpha_cg=ScalarUtil<T>::scalar_max) const{

	  assert_eq(D.size(), X.size());
	  assert_eq(_L.size(), X.size());
	  assert_eq(_H.size(), X.size());
	  T ret=ScalarUtil<T>::scalar_max;
	  T tmp;
#pragma omp parallel private(tmp)
	  {
		tmp=ScalarUtil<T>::scalar_max;
#pragma omp for
		for(size_t i=0;i<X.size();i++)
		  {
			if(D[i] > ScalarUtil<T>::scalar_eps && X[i] > _L[i])	//handle rounding err
			  tmp=std::min<T>(tmp,(X[i]-_L[i])/D[i]);
			else if(D[i] < -ScalarUtil<T>::scalar_eps && X[i] < _H[i])	//handle rounding err
			  tmp=std::min<T>(tmp,(X[i]-_H[i])/D[i]);
		  }

		OMP_CRITICAL_
		  ret=std::min<T>(ret,tmp);
	  }
	  return ret;
	}
	void project(const Vec& in,Vec& out) const{
	  assert_eq(in.size(), _L.size());
	  out.resize(_L.size());
	  OMP_PARALLEL_FOR_
		for(size_t i=0;i<in.size();i++)
		  out[i]=std::min<T>(std::max<T>(in[i],_L[i]),_H[i]);
	}
	void PHI(const Vec& in,Vec& out){
	  MASK_FACE(in,out,_face);
	}
	void BETA(const Vec& in,Vec& out, const Vec&phi){
	  
	  assert_eq(in.size(), _face.size());
	  out.resize(in.size());
	  OMP_PARALLEL_FOR_
		for(size_t i=0;i<in.rows();i++){
		  if(_face[i] == 0)
			out[i]=0.0f;
		  else if(_face[i] == 1)
			out[i]=std::max<T>(in[i],0.0f);
		  else 
			out[i]=std::min<T>(in[i],0.0f);
		}
	}
	void DECIDE_FACE(const Vec& x){

	  assert_eq(x.size(), _L.size());
	  assert_eq(x.size(), _H.size());
	  assert_eq(x.size(), _face.size());
	  const Vec &L = _L;
	  const Vec &H = _H;
	  _face.assign(x.rows(),0);
	  OMP_PARALLEL_FOR_
		for(size_t i=0;i<x.rows();i++)
		  if(abs(x[i]-L[i]) < ScalarUtil<T>::scalar_eps)
			_face[i]=2;
		  else if(abs(x[i]-H[i]) < ScalarUtil<T>::scalar_eps)
			_face[i]=1;
	}
	T PHITPHI(const Vec& x,const T&alphaBar,const Vec&phi){

	  assert_eq(x.size(), _L.size());
	  assert_eq(x.size(), _H.size());
	  assert_eq(x.size(), phi.size());
	  const Vec &L = _L;
	  const Vec &H = _H;
	  T phiTphi=0.0f;
#pragma omp parallel for reduction(+:phiTphi)
	  for(size_t i=0;i<x.rows();i++){
		T phiTilde=0.0f;
		if(phi[i] > 0.0f && x[i] > L[i])	//handle rounding error
		  phiTilde=std::min<T>((x[i]-L[i])/alphaBar,phi[i]);
		else if(phi[i] < 0.0f && x[i] < H[i])	//handle rounding error
		  phiTilde=std::max<T>((x[i]-H[i])/alphaBar,phi[i]);
		assert_ge(phiTilde*phi[i], 0.0f);
		phiTphi+=phiTilde*phi[i];
	  }
	  return phiTphi;
	}
	
  private:
	const Vec &_L;
	const Vec &_H;
	vector<char> _face;
  };

  // support plane constraints
  template <typename T>
  class PlaneProjector{

	typedef Eigen::Matrix<T,-1,1> Vec;
	typedef Eigen::Matrix<T,4,1> Vec4X;
	typedef Eigen::Matrix<T,3,1> Vec3X;
	typedef vector<Vec4X,Eigen::aligned_allocator<Vec4X> > VVec4X;
	typedef vector<VVec4X > VVVec4X;

  public:
    PlaneProjector(const VVVec4X &planes_for_each_node, const Vec &feasible_x):
	  _planes(planes_for_each_node), feasible_x(feasible_x){

	  assert_eq(_planes.size()*3, feasible_x.size());
	  updateConstraints();
	  assert_ext(isFeasible(_planes,feasible_x),"x = "<<feasible_x.transpose());
	}

	void updateConstraints(){
	  
	  const size_t x_size = _planes.size()*3;
	  _face.resize(x_size);
	  _face.assign(x_size,0);

	  _face_indices.resize(x_size/3);
	  for (int i = 0; i < _face_indices.size(); ++i){
		_face_indices[i].clear();
		_face_indices[i].reserve(3);
	  }
	}

	const vector<char> &getFace()const{
	  return _face;
	}

	const vector<vector<int> > &getFaceIndex()const{
	  return _face_indices;
	}

	// return the largest step in direction -D.
	T stepLimit(const Vec& X,const Vec& D, const T alpha_cg=ScalarUtil<T>::scalar_max) const{

	  const size_t num_points = _face_indices.size();
	  assert_eq(D.size(),num_points*3);
	  assert_eq(D.size(), X.size());

	  T alpha = alpha_cg+ScalarUtil<T>::scalar_eps;
	  for (size_t i = 0; i < num_points; ++i){

	  	const Vec3X di = D.template segment<3>(i*3);
	  	const Vec3X xi = X.template segment<3>(i*3);
		const VVec4X &p = _planes[i];
	  	for (size_t j = 0; j < p.size(); ++j){

	  	  const Vec3X nj = p[j].template segment<3>(0);
	  	  assert_in(nj.norm(),1.0f-ScalarUtil<T>::scalar_eps,1.0f+ScalarUtil<T>::scalar_eps);
	  	  const T nd = nj.dot(di);
	  	  if ( nd >= ScalarUtil<T>::scalar_eps){
	  		const T alpha_ij = (nj.dot(xi)+p[j][3])/nd;
			if (alpha_ij >= 0)
			  alpha = std::min<T>(alpha, alpha_ij);
	  	  }

		  if(dist(p[j],(xi-alpha_cg*di))<-ScalarUtil<double>::scalar_eps){
			alpha = std::min<T>(alpha, alpha_cg-ScalarUtil<double>::scalar_eps);
		  }

	  	}
	  }
	  assert_ge(alpha, 0.0f);
	  return alpha;
	}

	void project(const Vec& in,Vec& out) const{

	  const size_t num_points = _face_indices.size();
	  assert_eq(in.size(),num_points*3);
	  assert_ge(in.size(),3);
	  out.resize( in.size() );
	  Vec3X v = in.template segment<3>(0);
	  Vector3i aSet;
	  
	  for (int i = 0; i < in.size(); i += 3){

	  	aSet.setConstant(-1);
		findFeasible(_planes[i/3], feasible_x, i/3, v);
		bool found = findClosestPoint( _planes[i/3], in.template segment<3>(i), v, aSet );
		assert(found);
	  	out.template segment<3>(i) = v;
	  }
	  assert_ext(isFeasible(_planes,out),"x="<<in.transpose()<<"\nproject(x)="<<out.transpose());
	}

	void PHI(const Vec& in,Vec& out){

	  const size_t num_points = _face_indices.size();
	  assert_eq(in.size(),num_points*3);
	  out = in;
	  Vec3X temp;
	  temp.setZero();
	  for (int i = 0; i < in.size(); i += 3){
	  	assert_eq(_face[i], _face_indices[i/3].size());
	  	if (3 <= _face[i]){
	  	  out.template segment<3>(i).setZero();
	  	}else if (2 == _face[i]){
		  const int f0 = _face_indices[i/3][0];
		  const int f1 = _face_indices[i/3][1];
		  assert_ne(f0,f1);
		  const Vec3X n0 = _planes[i/3][f0].template segment<3>(0);
		  const Vec3X n1 = _planes[i/3][f1].template segment<3>(0);
		  const Vec3X n = n0.cross(n1);
		  out.template segment<3>(i) = in.template segment<3>(i).dot(n)*n;
	  	}else if (1 == _face[i]){
		  const Vec4X &p = _planes[i/3][_face_indices[i/3][0]];
	  	  projectToPlane(p, in.template segment<3>(i), temp);
	  	  out.template segment<3>(i) = temp;
	  	}
	  }
	  assert_eq(in,in);
	  assert_eq(out,out);
	}

	void BETA(const Vec& in, Vec& out, const Vec&phi){

	  const size_t num_points = _face_indices.size();
	  assert_eq(in.size(),num_points*3);
	  out.resize(in.size());
	  out.setZero();

	  Vec3X temp;
	  for (int i = 0; i < in.size(); i += 3){

	  	assert_eq(_face[i], _face_indices[i/3].size());
	  	if (1 == _face[i]){
	  	  const Vec3X n = _planes[i/3][_face_indices[i/3][0]].template segment<3>(0);
	  	  const T t = in.template segment<3>(i).dot(n);
	  	  if (t < 0)
	  		out.template segment<3>(i) = t*n;
	  	}else if (_face[i]>=2){
	  	  const bool found = findClosestPoint( _planes[i/3], _face_indices[i/3], in.template segment<3>(i), phi.template segment<3>(i), temp );
	  	  assert(found);
	  	  out.template segment<3>(i) = temp;
	  	}
	  }
	  assert_eq(phi,phi);
	}

	void DECIDE_FACE(const Vec& x){

	  const size_t num_points = _face_indices.size();
	  assert_eq(x.size(),num_points*3);
	  assert_eq(x,x);
	  for (int i = 0; i < num_points; i++ ){
	  	_face_indices[i].clear();
		Vec3X xi = x.template segment<3>(i*3);
	  	for (int f = 0; f < _planes[i].size(); ++f){
		  const T d = dist(_planes[i][f],xi);
	  	  if ( fabs(d) < ScalarUtil<T>::scalar_eps ){
	  		_face_indices[i].push_back(f);
		  }
	  	}
	  	_face[i*3] = _face_indices[i].size();
	  	_face[i*3+1] = _face[i*3];
	  	_face[i*3+2] = _face[i*3];
	  }
	}

	T PHITPHI(const Vec& x,const T&alphaBar,const Vec&phi){

	  assert_ext(isFeasible(_planes,x),"x="<<x.transpose());
	  assert_gt(alphaBar, ScalarUtil<T>::scalar_eps);
	  assert_eq(x.size() % 3,0);
	  assert_eq(x.size(), phi.size());
	  const Vec x_alpha_phi = x-alphaBar*phi;
	  Vec px;
	  project(x_alpha_phi, px);
	  const T phitphi = ((x-px).dot(phi))*(1.0/alphaBar);
	  assert_ge(phitphi,-ScalarUtil<T>::scalar_eps);
	  return phitphi>0.0f?phitphi:0.0f;
	}

	static VVVec4X &convert(const VVec4X &in, VVVec4X &out, const size_t num_nodes){
	  	 
	  out.resize(num_nodes);
	  for(size_t i = 0; i < num_nodes; i++)
		out[i] = in;
	  return out;
	}

  protected:
	void projectToPlane(const Vec4X &p, const Vec3X &in, Vec3X &out)const{

	  const Vec3X n = p.template segment<3>(0);
	  assert_in(n.norm(),1.0f-ScalarUtil<T>::scalar_eps,1.0f+ScalarUtil<T>::scalar_eps);
	  out = in-in.dot(n)*n;
	}
	
  private:
	const VVVec4X &_planes;
	const Vec feasible_x;
	vector<char> _face;
	vector<vector<int> > _face_indices;
  };

}//end of namespace

#endif /* _MPRGPPROJECTION_H_ */
