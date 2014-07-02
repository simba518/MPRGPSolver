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
	  _A(A), _B(B), _maxIterations(max_it), _toleranceFactor(tol),
	  _precond(precond), _projector(projector){
	  
	}
	
	int solve(Vec &x){
	  /// @todo
	  return 0;
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
	  
	  LowerBoundProjector<T,MAT> projector(L);
	  DiagonalInFacePreconSolver<T,MAT> precond(A, projector.getFace());
	  MPRGP<T, MAT, LowerBoundProjector<T,MAT>, DiagonalInFacePreconSolver<T,MAT> > solver(A, B, precond, projector, max_it, tol);
	  return solver.solve(x);
	}
  };

  template<typename T=double>
  class MPRGPBoxBound{

	typedef Eigen::Matrix<T,-1,1> Vec;
	
  public:
	template <typename MAT> 
	static int solve(const MAT &A,const Vec &B, const Vec &L, const Vec &U, Vec &x, const T tol=1e-3, const int max_it = 1000){

	  BoxBoundProjector<T,MAT> projector(L, U);
	  DiagonalInFacePreconSolver<T,MAT> precond(A, projector.getFace());
	  MPRGP<T, MAT, BoxBoundProjector<T,MAT>, DiagonalInFacePreconSolver<T,MAT> > solver(A, B, precond, projector, max_it, tol);
	  return solver.solve(x);
	}
  };
  
}//end of namespace
