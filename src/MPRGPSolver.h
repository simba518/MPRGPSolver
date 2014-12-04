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
	  _alphaBar=2.0f/specRad(_A,NULL,tol);
	  _iterationsOut = 0;
	  _residualOut = 0.0f;
	}
	
	int solve(Vec &result){

	  FUNC_TIMER();
	  
	  //declaration
	  T alphaCG,alphaF,beta;
	  Vec& AP=_gp;
	  Vec& AD=_gp;
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

		DEBUG_LOG(setprecision(16)<<"func: "<<_A.funcValue(result,_B));

		//test termination
		assert_eq(_g,_g);
		_projector.PHI(_g,_phi);
		_projector.BETA(_g,_beta,_phi);
		assert_eq(_phi.size(), _beta.size());
		_gp = _phi+_beta;
		_residualOut=_gp.norm();

		assert_le(_phi.dot(_beta),ScalarUtil<T>::scalar_eps*_residualOut);
		DEBUG_LOG(setprecision(10)<<"||g||: "<<_g.norm());
		DEBUG_LOG(setprecision(10)<<"||beta||: "<<_beta.norm());
		DEBUG_LOG(setprecision(10)<<"||phi||: "<<_phi.norm());
		DEBUG_LOG(setprecision(10)<<"residual: "<<_residualOut);
		DEBUG_LOG("phi: "<<_phi.transpose());
		DEBUG_LOG("beta: "<<_beta.transpose());

		// debug
		// DEBUG_FUN(assert(writeVTK(result, "beta_phi_g.vtk")));
		// DEBUG_FUN(assert(printFace()));

		if(_residualOut <= _toleranceFactor){
		  _iterationsOut = iteration;
		  result_code = 0;
		  break;
		}

		//test proportional x: beta*beta <= gamma*gamma*phi*phiTilde
		const T beta_norm = _beta.norm();
		assert_eq(beta_norm, beta_norm);
		assert_eq(_g,_g);

		if(beta_norm*beta_norm <= _Gamma*_Gamma*_projector.PHITPHI(result,_alphaBar,_phi)){

		  //prepare conjugate gradient
		  _A.multiply(_p,AP);
		  const T pd = _p.dot(AP);
		  assert_eq(pd,pd);
		  assert_gt(pd,0); // pd = p^t*A*p > 0
		  alphaCG = (_z.dot(_g)) / pd;
		  assert_eq(alphaCG, alphaCG);
		  assert_ge(alphaCG,0.0f);
		  alphaF = _projector.stepLimit(result,_p,alphaCG);
		  assert_eq(alphaF, alphaF);
		  assert_ge(alphaF,0.0f);

		  DEBUG_LOG("alphaCG: "<<alphaCG);
		  DEBUG_LOG("alphaF: "<<alphaF);
		  DEBUG_LOG("alphaF-alphaCG: "<<alphaF-alphaCG);

		  if(alphaCG <= alphaF){
			//conjugate gradient step
			DEBUG_LOG("cg step");
			num_cg ++;
			assert_ge(alphaCG,0.0f);
			result = result-alphaCG*_p;
			_g -= alphaCG*AP;

			assert_eq(_g,_g);
			_projector.PHI(_g, _phi);
			assert_ge(_g.dot(_phi),-ScalarUtil<T>::scalar_eps);
			_precond.solve(_phi,_z);
			assert_ge(_g.dot(_z),-ScalarUtil<T>::scalar_eps);
			beta = (_z.dot(AP)) / (_p.dot(AP));
			_p = _z-beta*_p;
			assert_eq(_p,_p);

		  }else{
			//expansion step
			DEBUG_LOG("exp step");
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
		  DEBUG_LOG("prop step");
		  num_prop ++;
		  assert_gt(beta_norm,0);
		  D = _beta;
		  _A.multiply(D,AD);
		  assert_gt(AD.norm(),0);
		  const T ddad = D.dot(AD);
		  assert_ne(ddad,0);
		  alphaCG = (_g.dot(D)) / ddad;
		  result -= alphaCG*D;

		  // @note we need to project result when 3D plane constraints are used.
		  xTmp = result;
		  _projector.project(xTmp,result);

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
		DEBUG_LOG(setprecision(10)<<"power delta: "<<delta);
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

	bool writeVTK(const Vec &x, const string filename)const{

	  Vec points(x.size()*4);
	  points.head(x.size()) = x;
	  points.segment(x.size(), x.size()) = _phi+x;
	  points.segment(x.size()*2, x.size()) = _beta+x;
	  points.segment(x.size()*3, x.size()) = _g+x;

	  ofstream out;
	  out.open(filename.c_str());
	  if (!out.is_open()){
		ERROR_LOG("failed to open this file: "<<filename);
		return false;
	  }

	  // head
	  out << "# vtk DataFile Version 3.1 \n";	  
	  out << "write phi(x), beta(x) and g(x) \nASCII\nDATASET UNSTRUCTURED_GRID\n";

	  // points
	  out << "POINTS "<< points.size()/3  <<" FLOAT\n";
	  for (int i = 0; i < points.size(); i += 3){
		out << points[i+0] << " " << points[i+1] << " " << points[i+2] << "\n";
	  }
	  
	  // lines
	  const int xp = x.size()/3;
	  const int num_lines = 3*xp;
	  out << "CELLS " <<num_lines<< " "<< 3*num_lines << "\n";
	  for (int i = 0; i < xp; i++){
		out << 2 << " "<< i << " "<< i+xp << "\n";
		out << 2 << " "<< i << " "<< i+2*xp << "\n";
		out << 2 << " "<< i << " "<< i+3*xp << "\n";
	  }
	  
	  out << "CELL_TYPES "<<num_lines<<"\n";
	  for (int i = 0; i < num_lines; ++i){
		out << 3;
		if (i == num_lines-1) out << "\n"; else  out << " ";
	  }

	  const bool succ = out.good();
	  out.close();
	  return succ;
	}
	bool printFace()const{
	  const vector<char> &f = _projector.getFace();
	  cout << "face: ";
	  for (int i = 0; i < f.size(); ++i)
		cout << (int)f[i] << " ";
	  cout << "\n";
	  return true;
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
	typedef Eigen::Matrix<T,3,1> Vec3X;
	typedef Eigen::Matrix<T,4,1> Vec4X;
	typedef vector<Vec4X,Eigen::aligned_allocator<Vec4X> > VVec4X;
	typedef vector<VVec4X > VVVec4X;
	
  public:
	template <typename MAT>
	static int solve(const MAT &A,const Vec &B, PlaneProjector<T> &projector, Vec &x, const T tol=1e-3, const int max_it = 1000){

	  assert_eq(A.rows(),B.size());
	  assert_eq(A.rows(),x.size());
	  DiagonalInFacePreconSolver<T,MAT> precond(A, projector.getFace());
	  MPRGP<T, MAT, PlaneProjector<T>, DiagonalInFacePreconSolver<T,MAT> > solver(A, B, precond, projector, max_it, tol);
	  const int rlst_code = solver.solve(x);
	  return rlst_code;
	}

	template <typename MAT>
	static int solve(const MAT &A,const Vec &B, const VVVec4X &planes_for_each_node, Vec &x, const T tol=1e-3, const int max_it = 1000){

	  PlaneProjector<T> projector(planes_for_each_node, x);
	  return solve(A, B, projector, x, tol, max_it);
	}

	template <typename MAT>
	static int solve(const MAT &A,const Vec &B, const VVec4X &planes, Vec &x, const T tol=1e-3, const int max_it = 1000){

	  VVVec4X planes_for_each_node;
	  PlaneProjector<T>::convert(planes, planes_for_each_node, x.size()/3);
	  return solve(A,B,planes_for_each_node, x, tol, max_it);
	}

	// load the problem from file, then solve it.
	static int solve(const string file_name, Vec&x, const T tol=1e-3, const int max_it = 1000){

	  SparseMatrix<T> A;
	  Vec B;
	  VVec4X planes;
	  int code = -1;
	  if (loadQP(A,B,planes,x,file_name)){
		for (int i = 0; i < planes.size(); ++i){
		  DEBUG_LOG("planes"<<i<<": "<<planes[i].transpose());
		}
		code = solve(FixedSparseMatrix<double>(A),B,planes,x,tol,max_it);
	  }
	  return code;
	}

	// compute the lagragian multipliers.
	static void computeLagMultipliers(const Vec &g, const VVVec4X &planes_for_each_node, 
									  const std::vector<std::vector<int> > &face_indices,
									  std::vector<std::vector<T> > &all_lambdas){
	  
	  const int num_verts = face_indices.size();
	  assert_eq(g.size(), num_verts*3);
	  all_lambdas.resize(num_verts);
	  for(int i = 0; i < num_verts; i++){
		const Vec3X gi = g.template segment<3>(i*3);
		computeLagMultipliers(gi, planes_for_each_node[i], face_indices[i], all_lambdas[i]);
	  }
	}

  protected:
	static void computeLagMultipliers(const Vec3X &gi, const VVec4X &planes,
									  const std::vector<int> &face_i,
									  std::vector<T> &lambdas){
	  
	  lambdas.resize(planes.size());
	  for (size_t i = 0; i < lambdas.size(); ++i){
		lambdas[i] = (T)0.0f;
	  }

	  if(face_i.size() == 1){

		const int p = face_i[0];
		assert_in(p, 0, (int)planes.size()-1);
		lambdas[p] = gi.dot(planes[p].template segment<3>(0));
		assert_ge(lambdas[p],0.0f);

	  }else if(face_i.size() == 2){

		const int p0 = face_i[0];
		const int p1 = face_i[1];
		assert_in(p0, 0, (int)planes.size()-1);
		assert_in(p1, 0, (int)planes.size()-1);

		Matrix<T, 3,2> N;
		N.template block<3,1>(0,0) = planes[p0].template segment<3>(0);
		N.template block<3,1>(0,1) = planes[p1].template segment<3>(0);
	
		const Matrix<T,2,2> A = (N.transpose()*N).inverse();
		assert_eq_ext(A, A, "N: " << N);
		const Matrix<T,2,1> la = A*(N.transpose()*gi);
		lambdas[p0] = la[0];
		lambdas[p1] = la[1];

		assert_ge(lambdas[p0],0.0f);
		assert_ge(lambdas[p1],0.0f);

	  }else if(face_i.size() >= 3){
	
		const int p0 = face_i[0];
		const int p1 = face_i[1];
		const int p2 = face_i[2];
		assert_in(p0, 0, (int)planes.size()-1);
		assert_in(p1, 0, (int)planes.size()-1);
		assert_in(p2, 0, (int)planes.size()-1);

		Matrix<T,3,3> N;
		N.template block<3,1>(0,0) = planes[p0].template segment<3>(0);
		N.template block<3,1>(0,1) = planes[p1].template segment<3>(0);
		N.template block<3,1>(0,2) = planes[p2].template segment<3>(0);
	
		const Matrix<T,3,3> A = (N.transpose()*N).inverse();
		assert_eq_ext(A, A, "N: " << N);
		const Vec3X la = A*(N.transpose()*gi);
		lambdas[p0] = la[0];
		lambdas[p1] = la[1];
		lambdas[p2] = la[2];

		assert_ge(lambdas[p0],0.0f);
		assert_ge(lambdas[p1],0.0f);
		assert_ge(lambdas[p2],0.0f);
	  }
	}

  };
  
}//end of namespace

#endif /* _MPRGPSOLVER_H_ */
