#ifndef _MPRGPPROJECTION_H_
#define _MPRGPPROJECTION_H_

#include <stdio.h>
#include <vector>
#include <eigen3/Eigen/Sparse>
#include <eigen3/Eigen/Dense>
#include <MPRGPUtility.h>
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
	T stepLimit(const Vec& X,const Vec& D) const{

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

	void BETA(const Vec& in,Vec& out){

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
		for(size_t i=0;i<x.size();i++)
		  if(abs(x[i]-L[i]) < ScalarUtil<T>::scalar_eps)
			_face[i]=-1;
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

	T stepLimit(const Vec& X,const Vec& D) const{

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
	void BETA(const Vec& in,Vec& out){
	  
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
			_face[i]=-1;
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

}//end of namespace

#endif /* _MPRGPPROJECTION_H_ */
