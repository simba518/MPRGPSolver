#ifndef _MPRGPSOLVER_H_
#define _MPRGPSOLVER_H_

#include "MPRGPPrecondition.h"
#include "MPRGPProjection.h"

namespace MATH{

  /**
   * @class MPRGP a framework for the MPRGP method, with
   * preconditioning and projecting method can be changed through the
   * template parameters.
   * 
   * Solve such problem:
   * min_{x} 1/2*x^t*A*x-x^t*B s.t. n_i*x_j+p_i>= 0
   * 
   */
  template <typename T, typename MAT, typename PROJECTOIN,typename PRECONDITION>
  class MPRGP{

	typedef Eigen::Matrix<T,-1,1> Vec;
	
  public:
	MPRGP(const MAT &A,const Vec &B,
		  PRECONDITION &precond, PROJECTOIN &projector,
		  const int max_it = 1000, const T tol=1e-3):
	  _A(A), _B(B), _precond(precond), _projector(projector){

	  setParameters(tol,max_it);
	  _Gamma=1.0f;
	  _alphaBar=2.0f/specRad(_A);
	  _iterationsOut = 0;
	  _residualOut = 0.0f;
	}
	
	int solve(Vec &result){

	  FUNC_TIMER();
	  
	  //declaration
	  T alphaCG,alphaF,beta;
	  Vec& AP=_gp;
	  Vec& AD=_gp;
	  Vec& y=_beta;	
	  Vec& xTmp=_beta;
	  Vec& D=_phi;

	  //initialization
	  _A.multiply(result,_g);
	  _g -= _B;
	  _projector.DECIDE_FACE(result);

	  assert_eq(_g,_g);
	  _projector.PHI(_g, _phi);
	  _precond.solve(_phi,_z);
	  _p = _z;
	  assert_eq(_p,_p);
	  int result_code = -1;
	  
	  int num_cg=0, num_exp=0, num_prop=0;

	  //MPRGP iteration
	  size_t iteration=0;
	  for(; iteration<_maxIterations; iteration++){

		DEBUG_LOG(setprecision(10)<<"func: "<<_A.funcValue(result,_B));

		//test termination
		assert_eq(_g,_g);
		_projector.PHI(_g,_phi);
		_projector.BETA(_g,_beta,_phi);
		assert_eq(_phi.size(), _beta.size());
		_gp = _phi+_beta;
		_residualOut=_gp.norm();
		INFO_LOG(setprecision(10)<<"residual: "<<_residualOut);
		if(_residualOut < _toleranceFactor){
		  _iterationsOut = iteration;
		  result_code = 0;
		  break;
		}

		//test proportional x: beta*beta <= gamma*gamma*phi*phiTilde
		const T beta_norm = _beta.norm();
		assert_eq(beta_norm, beta_norm);
		assert_eq(_g,_g);
		if(beta_norm*beta_norm <= _Gamma*_Gamma*_projector.PHITPHI(result,_alphaBar,_phi,_beta, _g)){

		  //prepare conjugate gradient
		  _A.multiply(_p,AP);
		  const T pd = _p.dot(AP);
		  assert_eq(pd,pd);
		  assert_gt(pd,0); // pd = p^t*A*p > 0
		  alphaCG = (_z.dot(_g)) / pd;
		  assert_eq(alphaCG, alphaCG);
		  assert_ge(alphaCG,0.0f);
		  y = result-alphaCG*_p;
		  alphaF = _projector.stepLimit(result,_p);
		  assert_eq(alphaF, alphaF);
		  assert_ge(alphaF,0.0f);
		  if(alphaCG <= alphaF){
			//conjugate gradient step
			INFO_LOG("cg step");
			num_cg ++;
			assert_ge(alphaCG,0.0f);
			result = y;
			_g -= alphaCG*AP;

			assert_eq(_g,_g);
			_projector.PHI(_g, _phi);
			assert_ge(_g.dot(_phi),0.0);
			_precond.solve(_phi,_z);
			assert_ge(_g.dot(_z),0.0);
			beta = (_z.dot(AP)) / (_p.dot(AP));
			_p = _z-beta*_p;
			assert_eq(_p,_p);

		  }else{
			//expansion step
			INFO_LOG("exp step");
			num_exp ++;
			xTmp = result-alphaF*_p;
			_g -= alphaF*AP;
			_projector.DECIDE_FACE(xTmp);
			assert_eq(_g,_g);
			_projector.PHI(_g, _phi);
			xTmp -= _alphaBar*_phi;
			_projector.project(xTmp,result);
			_A.multiply(result,_g);
			_g -= _B;
			_projector.DECIDE_FACE(result);

			assert_eq(_g,_g);
			_projector.PHI(_g, _phi);
			_precond.solve(_phi,_z);

			// _precond.solve(_g,_z);
			_p = _z;
			assert_eq(_p,_p);
		  }
		}else{
		  //proportioning
		  INFO_LOG("prop step");
		  num_prop ++;
		  assert_gt(beta_norm,0);
		  D = _beta;
		  _A.multiply(D,AD);
		  assert_gt(AD.norm(),0);
		  const T ddad = D.dot(AD);
		  assert_ne(ddad,0);
		  alphaCG = (_g.dot(D)) / ddad;
		  result -= alphaCG*D;
		  _g -= alphaCG*AD;
		  _projector.DECIDE_FACE(result);

		  assert_eq(_g,_g);
		  _projector.PHI(_g, _phi);
		  _precond.solve(_phi,_z);
		  _p = _z;
		  assert_eq(_p,_p);

		}
	  }

	  WARN_LOG_COND("MPRGP not convergent with "<< _maxIterations << " iterations."<<endl, iteration >= _maxIterations);
	  INFO_LOG("cg steps: "<< num_cg);
	  INFO_LOG("exp steps: "<< num_exp);
	  INFO_LOG("prop steps: "<< num_prop);
	  INFO_LOG("MPRGP iter: "<<iteration);
	  INFO_LOG("constraints: "<<COUNT_CONSTRAINTS(_projector.getFace()));

	  return result_code;
	}

	void setParameters(T toleranceFactor,size_t maxIterations) {
	  _maxIterations=maxIterations;
	  _toleranceFactor=toleranceFactor;
	  if(_toleranceFactor<1e-30f)
		_toleranceFactor=1e-30f;
	}
	size_t iterationsOut()const{return _iterationsOut;}
	T residualOut()const{return _residualOut;}
	static T specRad(const MAT& G,Vec* ev=NULL,const T& eps=1E-3f){

	  FUNC_TIMER()

	  T delta;
	  Vec tmp,tmpOut;
	  tmp.resize(G.rows());
	  tmpOut.resize(G.rows());
	  tmp.setRandom();
	  tmp.normalize();

	  T normTmpOut = 1.0;
	  //power method
	  // for(size_t iter=0;;iter++){ /// @todo
	  size_t iter=0;
	  for(;iter <= 1000;iter++){
		G.multiply(tmp,tmpOut);
		normTmpOut=tmpOut.norm();
		if(normTmpOut < ScalarUtil<T>::scalar_eps){
		  if(ev)*ev=tmp;
		  normTmpOut = ScalarUtil<T>::scalar_eps;
		  break;
		}
		tmpOut/=normTmpOut;
		delta=(tmpOut-tmp).norm();
		INFO_LOG(setprecision(10)<<"power delta: "<<delta);
		// printf("Power Iter %d Err: %f, SpecRad: %f\n",iter,delta,normTmpOut);
		if(delta <= eps){
		  if(ev)*ev=tmp;
		  break;
		}
		tmp=tmpOut;
	  }
	  INFO_LOG("power iter: "<<iter);
	  return normTmpOut;
	}

  protected:
	//problem
	const MAT& _A;
	const Vec& _B;

	//parameter
	size_t _maxIterations;
	T _toleranceFactor;
	T _Gamma, _alphaBar;

	// internal structures
	PRECONDITION &_precond;
	PROJECTOIN &_projector;
	
	//temporary
	vector<char> _face;
	Vec _g,_p,_z,_beta,_phi,_gp;
	size_t _iterationsOut;
	T _residualOut;
  };

  // a matrix providing matrix-vector production, rows, and diagonal elements.
  template <typename T>
  class FixedSparseMatrix{

  public:
	FixedSparseMatrix(const SparseMatrix<T> &M):A(M){
	  assert_eq(A.rows(),A.cols());
	  diag_A.resize(A.rows());
	  for(int k=0;k<A.outerSize();++k)
		for(typename Eigen::SparseMatrix<T>::InnerIterator it(A,k);it;++it){
		  if (it.col() == it.row()){
			assert_eq(it.row(),k);
			diag_A[k] = it.value();
			break;
		  }
		}
	}
	
	// result = A*x
	template <typename VEC,typename VEC_OUT>
	void multiply(const VEC& x,VEC_OUT& result)const{
	  assert_eq(A.cols(), x.size());
	  result = A*x;
	}
	
	// return 1/2*x^t*A*x-x^t*b
	template <typename VEC, typename VEC_SECOND>
	T funcValue(const VEC& x,VEC_SECOND&b)const{
	  assert_eq(A.rows(),x.size());
	  assert_eq(A.cols(),x.size());
	  assert_eq(b.size(),x.size());
	  return 0.5f*x.dot(A*x)-x.dot(b);
	}

	int rows()const{
	  return A.rows();
	}

	double diag(const int i)const{
	  assert_in(i,0,diag_A.size()-1);
	  return diag_A[i];
	}

  protected:
	const SparseMatrix<T> &A;
	Matrix<T,-1,1> diag_A;
  };

  // solvers
  template<typename T=double>
  class MPRGPLowerBound{

	typedef Eigen::Matrix<T,-1,1> Vec;
	
  public:
	template <typename MAT>
	static int solve(const MAT&A,const Vec&B,const Vec&L,Vec &x,const T tol=1e-3,const int max_it=1000){
	  
	  LowerBoundProjector<T> projector(L);
	  DiagonalInFacePreconSolver<T,MAT> precond(A, projector.getFace());
	  MPRGP<T, MAT, LowerBoundProjector<T>, DiagonalInFacePreconSolver<T,MAT> > solver(A, B, precond, projector, max_it, tol);
	  return solver.solve(x);
	}
  };

  template<typename T=double>
  class MPRGPBoxBound{

	typedef Eigen::Matrix<T,-1,1> Vec;
	
  public:
	template <typename MAT> 
	static int solve(const MAT &A,const Vec &B, const Vec &L, const Vec &U, Vec &x, const T tol=1e-3, const int max_it = 1000){

	  BoxBoundProjector<T> projector(L, U);
	  DiagonalInFacePreconSolver<T,MAT> precond(A, projector.getFace());
	  MPRGP<T, MAT, BoxBoundProjector<T>, DiagonalInFacePreconSolver<T,MAT> > solver(A, B, precond, projector, max_it, tol);
	  const int rlst_code = solver.solve(x);
	  return rlst_code;
	}
  };

  template<typename T=double>
  class MPRGPPlane{

	typedef Eigen::Matrix<T,-1,1> Vec;
	typedef Eigen::Matrix<T,4,1> Vec4X;
	typedef vector<Vec4X,Eigen::aligned_allocator<Vec4X> > VVec4X;
	
  public:
	template <typename MAT>
	static int solve(const MAT &A,const Vec &B, const VVec4X &planes, Vec &x, const T tol=1e-3, const int max_it = 1000){

	  assert_eq(A.rows(),B.size());
	  assert_eq(A.rows(),x.size());
	  PlaneProjector<T> projector(planes,x.size());
	  DiagonalInFacePreconSolver<T,MAT> precond(A, projector.getFace());
	  MPRGP<T, MAT, PlaneProjector<T>, DiagonalInFacePreconSolver<T,MAT> > solver(A, B, precond, projector, max_it, tol);
	  const int rlst_code = solver.solve(x);
	  return rlst_code;
	}

	// load the problem from file, then solve it.
	static int solve(const string file_name, Vec&x, const T tol=1e-3, const int max_it = 1000){

	  SparseMatrix<T> A;
	  Vec B;
	  VVec4X planes;
	  int code = -1;
	  if (loadQP(A,B,planes,x,file_name))
		code = solve(FixedSparseMatrix<double>(A),B,planes,x,tol,max_it);
	  return code;
	}

  };
  
}//end of namespace

#endif /* _MPRGPSOLVER_H_ */
