#ifndef _MPRGPSOLVER_H_
#define _MPRGPSOLVER_H_

#include "MPRGPPrecondition.h"
#include "MPRGPProjection.h"

namespace MATH{

  /**
   * @class MPRGP a framework for the MPRGP method, with
   * preconditioning and projecting method can be changed through the
   * template parameters.
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

	  _projector.PHI(_g, _phi);
	  _precond.solve(_phi,_z);
	  // _precond.solve(_g,_z);
	  _p = _z;
	  int result_code = -1;

	  //MPRGP iteration
	  size_t iteration=0;
	  for(; iteration<_maxIterations; iteration++){

		//test termination
		_projector.PHI(_g,_phi);
		_projector.BETA(_g,_beta,_phi);
		assert_eq(_phi.size(), _beta.size());
		_gp = _phi+_beta;
		_residualOut=_gp.norm();
		if(_residualOut < _toleranceFactor){
		  _iterationsOut=iteration;
		  result_code = 0;
		  break;
		}

		//test proportional x: beta*beta <= gamma*gamma*phi*phiTilde
		const T beta_norm = _beta.norm();
		if(beta_norm*beta_norm <= _Gamma*_Gamma*_projector.PHITPHI(result,_alphaBar,_phi,_beta, _g)){

		  //prepare conjugate gradient
		  _A.multiply(_p,AP);
		  alphaCG = (_z.dot(_g)) / (_p.dot(AP));
		  y = result-alphaCG*_p;
		  alphaF = _projector.stepLimit(result,_p);
		  
		  if(alphaCG <= alphaF){

			//conjugate gradient step
			result = y;
			_g -= alphaCG*AP;

			_projector.PHI(_g, _phi);
			_precond.solve(_phi,_z);
			// _precond.solve(_g,_z);
			beta = (_z.dot(AP)) / (_p.dot(AP));
			_p = _z-beta*_p;

		  }else{
			
			//expansion step
			xTmp = result-alphaF*_p;
			_g -= alphaF*AP;
			_projector.DECIDE_FACE(xTmp);
			_projector.PHI(_g, _phi);
			xTmp -= _alphaBar*_phi;
			_projector.project(xTmp,result);
			_A.multiply(result,_g);
			_g -= _B;
			_projector.DECIDE_FACE(result);

			_projector.PHI(_g, _phi);
			_precond.solve(_phi,_z);

			// _precond.solve(_g,_z);
			_p = _z;

		  }
		}else{
		  
		  //proportioning
		  D = _beta;
		  _A.multiply(D,AD);
		  alphaCG = (_g.dot(D)) / (D.dot(AD));
		  result -= alphaCG*D;
		  _g -= alphaCG*AD;
		  _projector.DECIDE_FACE(result);

		  _projector.PHI(_g, _phi);
		  _precond.solve(_phi,_z);
		  // _precond.solve(_g,_z);
		  _p = _z;

		}
	  }

	  if (iteration >= _maxIterations){
		cout << "not convergent with "<< _maxIterations << " iterations."<<endl;
	  }
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

	  T delta;
	  Vec tmp,tmpOut;
	  tmp.resize(G.rows());
	  tmpOut.resize(G.rows());
	  tmp.setRandom();
	  tmp.normalize();

	  //power method
	  // for(size_t iter=0;;iter++){ /// @todo
	  for(size_t iter=0;iter <= 1000;iter++){
		G.multiply(tmp,tmpOut);
		T normTmpOut=tmpOut.norm();
		if(normTmpOut < ScalarUtil<T>::scalar_eps){
		  if(ev)*ev=tmp;
		  return ScalarUtil<T>::scalar_eps;
		}
		tmpOut/=normTmpOut;
		delta=(tmpOut-tmp).norm();
		// printf("Power Iter %d Err: %f, SpecRad: %f\n",iter,delta,normTmpOut);
		if(delta <= eps){
		  if(ev)*ev=tmp;
		  return normTmpOut;
		}
		tmp=tmpOut;
	  }
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
	template <typename VEC,typename VEC_OUT>
	void multiply(const VEC& x,VEC_OUT& result)const{
	  assert_eq(A.cols(), x.size());
	  result = A*x;
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
  };
  
}//end of namespace

#endif /* _MPRGPSOLVER_H_ */
