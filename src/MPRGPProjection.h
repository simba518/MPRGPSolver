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
	  _face.assign(_L.size(),char(0));
	}
	const vector<char> &getFace()const{
	  return _face;
	}

	// return the largest step in direction -D.
	T stepLimit(const Vec&X,const Vec&D,const T alpha_cg=ScalarUtil<T>::scalar_max)const{

	  assert_eq(D.size(), X.size());
	  assert_eq(_L.size(), X.size());
	  T ret=ScalarUtil<T>::scalar_max;
	  T tmp;
// #pragma omp parallel private(tmp)
	  {
		tmp=ScalarUtil<T>::scalar_max;
// #pragma omp for
		for(size_t i=0;i<X.size();i++){
		  if(D[i] > ScalarUtil<T>::scalar_eps && X[i] > _L[i])	//handle rounding err
			tmp=std::min<T>(tmp,(X[i]-_L[i])/D[i]);
		}

		// OMP_CRITICAL_
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
	  _face.assign(x.size(),char(0));
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
// #pragma omp parallel for reduction(+:phiTphi)
	  for(size_t i=0;i<x.rows();i++){
		T phiTilde=0.0f;
		if(phi[i] > 0.0f && x[i] > L[i])	//handle rounding error
		  phiTilde=std::min<T>((x[i]-L[i])/alphaBar,phi[i]);
		assert_ge(phiTilde*phi[i], 0.0f);
		phiTphi+=phiTilde*phi[i];
	  }
	  return phiTphi;
	}

	bool isFeasible(const Vec &x)const{

	  OMP_PARALLEL_FOR_
		for(size_t i=0; i < x.size();i++){
		  if( x[i] < _L[i]-ScalarUtil<T>::scalar_eps )
			return false;
		}
	  return true;
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
	  _face.assign(_L.size(), char(0));
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
// #pragma omp parallel private(tmp)
	  {
		tmp=ScalarUtil<T>::scalar_max;
// #pragma omp for
		for(size_t i=0;i<X.size();i++)
		  {
			if(D[i] > ScalarUtil<T>::scalar_eps && X[i] > _L[i])	//handle rounding err
			  tmp=std::min<T>(tmp,(X[i]-_L[i])/D[i]);
			else if(D[i] < -ScalarUtil<T>::scalar_eps && X[i] < _H[i])	//handle rounding err
			  tmp=std::min<T>(tmp,(X[i]-_H[i])/D[i]);
		  }

		// OMP_CRITICAL_
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
	  _face.assign(x.rows(),char(0));
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
// #pragma omp parallel for reduction(+:phiTphi)
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

	bool isFeasible(const Vec &x)const{

	  OMP_PARALLEL_FOR_
		for(size_t i=0; i < x.size();i++){
		   /// @bug lsw, not test
		  if( x[i] < _L[i]-ScalarUtil<T>::scalar_eps ||x[i]>_H[i]+ScalarUtil<T>::scalar_eps)
			return false;
		}
	  return true;
	}
	
  private:
	const Vec &_L;
	const Vec &_H;
	vector<char> _face;
  };

  // support plane constraints: n.dot(x) + p >= 0, where n and x \in R^3.
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
	  assert_ext(isFeasible(feasible_x),"x = "<<feasible_x.transpose());
	}

	void updateConstraints(){
	  
	  const size_t x_size = _planes.size()*3;
	  _face.resize(x_size);
	  _face.assign(x_size,char(0));

	  _face_indices.resize(x_size/3);
	  for (int i = 0; i < _face_indices.size(); ++i){
		_face_indices[i].clear();
		_face_indices[i].reserve(3);
	  }
	}

	const vector<char> &getFace()const{
	  return _face;
	}

	const VVVec4X &getPlanes()const{
	  return _planes;
	}

	const vector<vector<int> > &getFaceIndex()const{
	  return _face_indices;
	}

	// return the largest step in direction -D.
	T stepLimit(const Vec&X,const Vec&D,const T alpha_cg=ScalarUtil<T>::scalar_max)const{

	  const size_t num_points = _face_indices.size();
	  assert_eq(D.size(),num_points*3);
	  assert_eq(D.size(), X.size());
	  assert_ext(isFeasible(X), "alpha_cg: "<<alpha_cg);

	  T alpha = (alpha_cg == ScalarUtil<T>::scalar_max ? alpha_cg: (alpha_cg+ScalarUtil<T>::scalar_eps));
	  assert_gt(alpha, 0.0);
	  for (size_t i = 0; i < num_points; ++i){

	  	const Vec3X di = D.template segment<3>(i*3);
	  	const Vec3X xi = X.template segment<3>(i*3);
		const VVec4X &p = _planes[i];
	  	for (size_t j = 0; j < p.size(); ++j){
		  alpha = stepLimit(p[j], di, xi, alpha, alpha_cg);
	  	}
	  }
	  assert_ge(alpha, 0.0f);
	  assert_ext(isFeasible(X-alpha*D), "alpha: "<<alpha<<", alpha_cg: "<<alpha_cg);
	  return alpha;
	}

	void project(const Vec& in,Vec& out) const{

	  assert_eq(in.size(),_face_indices.size()*3);
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
	  assert_ext(isFeasible(out),"x="<<in.transpose()<<"\nproject(x)="<<out.transpose());
	}

	void PHI(const Vec& in,Vec& out){

	  assert_eq(in.size(),_face_indices.size()*3);
	  out = in;
	  Vec3X temp;
	  temp.setZero();
	  for (int i = 0; i < in.size(); i += 3){
	  	assert_eq(_face[i], _face_indices[i/3].size());
	  	if (2 == _face[i]){
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
		if (3 <= _face[i] || out.template segment<3>(i).dot(in.template segment<3>(i)) < 0 ){
		  // blocked or rounding error: gradient is almost perpendicular to the plane.
		  out.template segment<3>(i).setZero();
		}
	  }
	  assert_eq(in,in);
	  assert_eq(out,out);
	}

	void BETA(const Vec& in, Vec& out, const Vec&phi){

	  assert_eq(in.size(),_face_indices.size()*3);
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

	  // assert_ext(isFeasible(x),"x="<<x.transpose());
	  assert_gt(alphaBar, ScalarUtil<T>::scalar_eps);
	  assert_eq(x.size() % 3,0);
	  assert_eq(x.size(), phi.size());
	  const Vec x_alpha_phi = x-alphaBar*phi;
	  Vec px;
	  project(x_alpha_phi, px);
	  const T phitphi = ((x-px).dot(phi))*(1.0/alphaBar);
	  // assert_ge(phitphi,-ScalarUtil<T>::scalar_eps);
	  return phitphi>0.0f?phitphi:0.0f;
	}

	bool isFeasible(const Vec &x)const{
	  return MATH::isFeasible(this->getPlanes(), x);
	}

  protected:
	inline void projectToPlane(const Vec4X &p, const Vec3X &in, Vec3X &out)const{

	  const Vec3X n = p.template segment<3>(0);
	  assert_in(n.norm(),1.0f-ScalarUtil<T>::scalar_eps,1.0f+ScalarUtil<T>::scalar_eps);
	  out = in-in.dot(n)*n;
	}
	inline T stepLimit(const Vec4X &p, const Vec3X &d, const Vec3X &x, T alpha, const T alpha_cg)const{
	  
	  const T nd = p.template segment<3>(0).dot(d);
	  const T dnorm = d.norm();
	  const T tol = ScalarUtil<T>::scalar_eps*dnorm*dnorm;

	  if ( nd >= tol){
	  	const T alpha_ij = (p.template segment<3>(0).dot(x)+p[3])/nd;
	  	if (alpha_ij >= 0)
	  	  alpha = std::min<T>(alpha, alpha_ij);
	  }
	  const T tol_diff = -ScalarUtil<T>::scalar_eps;
	  while( alpha >= ScalarUtil<T>::scalar_eps && dist( p,(x-alpha*d) ) < tol_diff){
	  	alpha *= 0.5f;
	  }
	  return alpha;
	}
	
  private:
	const VVVec4X &_planes;
	const Vec feasible_x;
	vector<char> _face;
	vector<vector<int> > _face_indices;
  };

  // support decoupled constraints: J*x >= c, where J*J^t is diagonal.
  template <typename T>
  class DecoupledConProjector{

	typedef Eigen::Matrix<T,-1,1> Vec;
	
  public:
	DecoupledConProjector(const SparseMatrix<T> &J, const Vec &JJt, const Vec &c):
	  J(J),JJt(JJt),c(c){

	  assert_eq(J.rows(), JJt.size());
	  assert_eq(c.size(), J.rows());
	  face.resize(J.rows());
	  face.assign(face.size(),char(0));
	}
	const vector<char> &getFace()const{
	  return face;
	}
	const SparseMatrix<T> &getConMatrix()const{
	  return J;
	}

	// return the largest step in direction -D.
	T stepLimit(const Vec &X,const Vec&D,const T alpha_cg=ScalarUtil<T>::scalar_max)const{

	  T alpha = (alpha_cg == ScalarUtil<T>::scalar_max ? alpha_cg: (alpha_cg+ScalarUtil<T>::scalar_eps));
	  assert_gt(alpha, 0.0);
	  assert_eq(X.size(), D.size());
	  assert_eq(J.cols(), X.size());
	  
	  const Vec Jx = J*X;
	  const Vec Jd = J*D;
	  for (int i = 0; i < Jd.size(); ++i){
		if (Jd[i] > ScalarUtil<T>::scalar_eps && Jx[i] > c[i]){
		  const T ti = (Jx[i]-c[i])/Jd[i];
		  assert_eq(ti,ti);
		  alpha = std::min<T>(alpha, ti);
		}
	  }

	  assert_ge(alpha, 0.0f);
	  return alpha;
	}
	void project(const Vec &x,Vec &y) const{

	  assert_eq(x.size(), J.cols());
	  Vec lambda = c - J*x;
	  OMP_PARALLEL_FOR_
		for(int i = 0; i < lambda.size(); i++){
		  assert_ge(JJt[i], ScalarUtil<T>::scalar_eps);
		  lambda[i] = std::max<T>(lambda[i]/JJt[i], 0.0);
		}
	  y = x + J.transpose()*lambda;
	  assert_ext(isFeasible(y),"Jy-c:\n"<<(J*y-c).transpose());
	}
	void PHI(const Vec &g,Vec &phi) const{

	  Vec lambda = J*g;
	  assert_eq(J.cols(), g.size());
	  assert_eq(lambda.size(), (int)face.size());
	  for (size_t i = 0; i < face.size(); ++i){
		if(0 == face[i]){
		  lambda[i] = 0;
		}else{
		  lambda[i] = -lambda[i]/JJt[i];
		}
	  }
	  phi = g+J.transpose()*lambda;
	}
	void BETA(const Vec &g, Vec &beta, const Vec &phi){

	  beta = g-phi;
	  Vec lambda = J*beta;
	  for (size_t i = 0; i < face.size(); ++i){
	  	if(0 != face[i])
	  	  lambda[i] = std::max<T>(0.0,lambda[i]/JJt[i]);
	  }
	  beta -= J.transpose()*lambda;
	}
	void DECIDE_FACE(const Vec& x){

	  assert_eq(x.size(), J.cols());
	  assert_eq(c.size(), J.rows());
	  face.assign(c.size(),char(0));
	  const Vec Jx = J*x;
	  OMP_PARALLEL_FOR_
		for(int i = 0; i < c.size(); i++){
		  if(abs(Jx[i]-c[i]) < ScalarUtil<T>::scalar_eps)
			face[i] = 1;
		}
	}
	T PHITPHI(const Vec &x, const T alpha_bar, const Vec &phi) const{

	  assert_gt(alpha_bar, ScalarUtil<T>::scalar_eps);
	  assert_eq(x.size(), phi.size());
	  const Vec x_alpha_phi = x-alpha_bar*phi;
	  Vec px;
	  project(x_alpha_phi, px);
	  const T phitphi = ((x-px).dot(phi))*(1.0/alpha_bar);
	  return phitphi > 0.0 ? phitphi : 0.0;
	}

	bool isFeasible(const Vec &x)const{

	  assert_eq(x.size(), J.cols());
	  assert_eq(c.size(), J.rows());
	  const Vec Jx = J*x;
	  OMP_PARALLEL_FOR_
		for(int i = 0; i < c.size(); i++){
		  if( Jx[i] < c[i] - ScalarUtil<T>::scalar_eps){
			return false;
		  }
		}
	  return true;
	}
	
  private:
	const SparseMatrix<T> &J;
	const Vec &JJt; // diagonal elements of J*J^t
	const Vec &c;
	vector<char> face;
  };

}//end of namespace

#endif /* _MPRGPPROJECTION_H_ */
