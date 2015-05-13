#ifndef _FLEXIBLEPCG_H_
#define _FLEXIBLEPCG_H_

#include <string>
#include "MPRGPPrecondition.h"

namespace MATH{

  class IdentityPrecond{
  
  public:
	template<class VEC>
	void solve(const VEC &r, VEC &z, const VEC &phi)const{
	  z = r;
	}
  };
  
  /**
   * @class FlexiblePCG 
   * @see http://en.wikipedia.org/wiki/Conjugate_gradient_method#The_flexible_preconditioned_conjugate_gradient_method
   */
  template<class T, class MAT, class VEC, class PRECONDITIONER=IdentityPrecond, bool FLEXIBLE=true>
  class FlexiblePCG{
	
  public:
	FlexiblePCG(const MAT &A,PRECONDITIONER &precond,const T tol=1e-3, const int max_it=1e3):
	  A(A), precond(precond), tol(tol), max_it(max_it), name("FlexiblePCG"){}

	int solve(const VEC &b, VEC &x){
	  
	  r = b-A*x;
	  precond.solve(r, z, r);
	  p = z;

	  debug_fun(printFuncValue(A,b,x));
	  const T residual = r.norm();
	  DEBUG_LOG(name << " residual = " << residual);
	  if(residual <= tol)
		return 0;
	  
	  if (FLEXIBLE)
		r0 = r;

	  int it = 0;
	  for (; it < max_it; ++it){

		Ap = A*p;
		const T z0xr0 = z.dot(r);
		const T alpha = z0xr0/p.dot(Ap);
		x += alpha*p;
		debug_fun(printFuncValue(A,b,x));
		r -= alpha*Ap;
		const T residual = r.norm();
		DEBUG_LOG(name << " residual = " << residual);
		if(residual <= tol)
		  break;
	
		precond.solve(r, z, r);
		T beta = 0;
		if (FLEXIBLE){
		  beta = z.dot(r-r0)/z0xr0;
		  r0 = r;
		}else{
		  beta = z.dot(r)/z0xr0;
		}
		p = z+beta*p;
	  }

	  INFO_LOG(name << " iterations: "<< it);
	  return it<=max_it ? 0:-1;
	}

	void setName(const std::string name){
	  this->name = name;
	}

	void printFuncValue(const MAT& A, const VEC &b, const VEC &x)const{
	  const T f = (x.transpose().dot(A*x)*0.5-x.transpose().dot(b));
	  cout << name << setprecision(12) << " func = " << f  << endl;
	}
			
  private:
	const MAT &A;
	PRECONDITIONER &precond;

	T tol;
	int max_it;
	VEC r, z, p, Ap, r0;
	std::string name;
  };

}//end of namespace

#endif /*_FLEXIBLEPCG_H_*/
