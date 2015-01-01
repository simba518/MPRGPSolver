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
  class MPRGPBase{

	typedef Eigen::Matrix<T,-1,1> Vec;
	
  public:
	MPRGPBase(const MAT &A,const Vec &B,
			  PRECONDITION &precond, PROJECTOIN &projector,
			  const int max_it = 1000, const T tol=1e-3):
	  A(A), B(B), precond(precond), projector(projector) {

	  setParameters(tol,max_it);
	  gamma=1.0f;
	  alpha_bar=2.0/specRad(A,NULL,tol);
	  iterations_out = 0;
	  residual_out = 0.0f;
	  DEBUG_LOG(setprecision(16) << "alpha_bar: " << alpha_bar);
	}
	
	void setParameters(T toleranceFactor,size_t maxIterations) {
	  max_iterations=maxIterations;
	  tolerance_factor=toleranceFactor;
	  if(tolerance_factor<1e-30f)
		tolerance_factor=1e-30f;
	}
	size_t iterationsOut()const{return iterations_out;}
	T residualOut()const{return residual_out;}
	static T specRad(const MAT& G,Vec* ev=NULL,const T& eps=1E-3f){

	  FUNC_TIMER_CLASS timer("compute spectral radius");

	  T delta;
	  Vec tmp,tmpOut;
	  tmp.resize(G.rows());
	  tmpOut.resize(G.rows());
	  tmp.setRandom(); ///@todo warm start ?
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
		// DEBUG_LOG(setprecision(10)<<"power delta: "<<delta);
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
	  points.segment(x.size(), x.size()) = phi+x;
	  points.segment(x.size()*2, x.size()) = beta+x;
	  points.segment(x.size()*3, x.size()) = g+x;

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
	  const vector<char> &f = projector.getFace();
	  cout << "face: ";
	  for (int i = 0; i < f.size(); ++i)
		cout << (int)f[i] << " ";
	  cout << "\n";
	  return true;
	}
	void printSolveInfo()const{
	  INFO_LOG("cg steps: "<< num_cg);
	  INFO_LOG("exp steps: "<< num_exp);
	  INFO_LOG("prop steps: "<< numprop);
	  INFO_LOG("MPRGP iter: "<<iteration);
	  INFO_LOG("constraints: "<<COUNT_CONSTRAINTS(projector.getFace()));
	}

  protected:
	inline void initialize(const Vec &result){
	  
	  A.multiply(result,g);
	  g -= B;
	  projector.DECIDE_FACE(result);
	  projector.PHI(g, phi);
	  precond.solve(g,z, phi);     assert_eq(z,z);
	  p = z;
	  
	  result_code = -1;
	  num_cg = num_exp = numprop = iteration = 0;

	  DEBUG_FUN(fun_val = A.funcValue(result, B); cout<< setprecision(12) << "func = " << fun_val<<endl;);
	}
	inline T computeGradients(const Vec &g, const bool comp_phi = true){

	  // DEBUG_FUN(assert(printFace()));
	  if (comp_phi)
		projector.PHI(g,phi);
	  projector.BETA(g,beta,phi);
	  gp = phi+beta;
	  const T residual = gp.norm();
	  assert_le(phi.dot(beta),ScalarUtil<T>::scalar_eps*residual);

	  // DEBUG_LOG(setprecision(10)<<"||g||: "<<g.norm());
	  // DEBUG_LOG(setprecision(10)<<"||beta||: "<<beta.norm());
	  // DEBUG_LOG(setprecision(10)<<"||phi||: "<<phi.norm());
	  // DEBUG_LOG(setprecision(10)<<"residual: "<<residual);
	  // DEBUG_LOG("g: "<<g.transpose());
	  // DEBUG_LOG("phi: "<<phi.transpose());
	  // DEBUG_LOG("beta: "<<beta.transpose());

	  return residual;
	}
	inline bool proportional(const Vec &result)const{
	  
	  //test proportional x: beta*beta <= gamma*gamma*phi*phiTilde
	  const T beta_norm = beta.norm();
	  const T left = beta_norm*beta_norm;  assert_eq(left, left);
	  const T phitphi = projector.PHITPHI(result,alpha_bar,phi);
	  const T right = gamma*gamma*phitphi; assert_eq(right, right);
	  return left <= right;
	}
	inline void CGStep(const Vec &AP, T alpha_cg, Vec &result){

	  DEBUG_LOG("cg_step");
	  num_cg ++;

	  assert_eq(alpha_cg, alpha_cg);
	  result -= alpha_cg*p;
	  g -= alpha_cg*AP;
	  projector.PHI(g, phi);  assert_ge(g.dot(phi),0);
	  precond.solve(g,z, phi);   assert_eq(z, z);  assert_ge(g.dot(z),0);
	  const T beta = (z.dot(AP)) / (p.dot(AP)); assert_eq(beta, beta);
	  p = z-beta*p;
	  DEBUG_FUN(fun_val = A.funcValue(result, B); cout<< setprecision(12) << "func = " << fun_val<<endl;);
	}
	inline void ExpStep(const Vec &AP, T alpha_f, Vec &result){

	  DEBUG_LOG("exp_step");
	  num_exp ++;

	  assert_eq(alpha_f, alpha_f);
	  Vec &xTmp=beta;
	  xTmp = result-alpha_f*p;
	  g -= alpha_f*AP;
	  projector.DECIDE_FACE(xTmp);
	  projector.PHI(g, phi);
	  xTmp -= alpha_bar*phi;
	  projector.project(xTmp,result);
	  A.multiply(result,g);
	  g -= B;
	  projector.DECIDE_FACE(result);
	  projector.PHI(g, phi); 
	  precond.solve(g,z, phi); assert_eq(z, z);
	  p = z;
	  DEBUG_FUN(fun_val = A.funcValue(result, B); cout<< setprecision(12) << "func = " << fun_val<<endl;);
	}
	inline void PropStep(Vec &result){

	  DEBUG_LOG("prop_step");
	  numprop ++;

	  const Vec &D = beta;
	  Vec& AD=gp;
	  A.multiply(D,AD);
	  const T ddad = D.dot(AD); assert_ne(ddad,0);
	  const T alpha_cg = (g.dot(D)) / ddad; assert_eq(alpha_cg, alpha_cg);
	  result -= alpha_cg*D;

	  Vec& xTmp=beta;
	  xTmp = result;
	  projector.project(xTmp,result);
	  g -= alpha_cg*AD;
	  projector.DECIDE_FACE(result);
	  projector.PHI(g, phi);
	  precond.solve(g,z, phi); assert_eq(z, z);
	  p = z;
	  DEBUG_FUN(fun_val = A.funcValue(result, B); cout<< setprecision(12) << "func = " << fun_val<<endl;);
	}

	inline T projectFunValue(const Vec &x)const{
	  static Vec tempx; /// @todo
	  projector.project(x, tempx);
	  return A.funcValue(tempx, B);
	}
	inline void ExpMonotonicStep(Vec &result){

	  DEBUG_LOG("exp_step");
	  num_exp ++;

	  projector.project(result, y);
	  A.multiply(y,g);
	  g -= B;
	  projector.DECIDE_FACE(y);
	  projector.PHI(g, phi);
	  y -= alpha_bar*phi;
	  projector.project(y, result);

	  A.multiply(result,g);
	  g -= B;
	  projector.DECIDE_FACE(result);
	  projector.PHI(g, phi); 
	  precond.solve(g,z, phi); assert_eq(z, z);
	  p = z;

	  DEBUG_FUN(fun_val = A.funcValue(result, B); cout<< setprecision(12) << "func = " << fun_val<<endl;);
	}
	inline T CGMonotonicStep(Vec &result, bool &result_is_prop){

	  Vec &AP=gp;
	  A.multiply(p, AP);
	  const T pd = p.dot(AP); assert_ge(pd, 0); // pd = p^t*A*p > 0
	  T alpha_cg = (z.dot(g)) / pd; assert_ge(alpha_cg, 0);

	  y = result - alpha_cg*p;
	  T fy = projectFunValue(y);
	  T fx = A.funcValue(result, B);
	  assert( isFeasible(projector.getPlanes(), result) );

	  result_is_prop = true;
	  while( result_is_prop && residual_out > tolerance_factor
			 && fy<=fx && iteration < max_iterations ){

		DEBUG_FUN(fun_val = fy; cout<< setprecision(12) << "func = " << fun_val<<endl;);
		result = y;
		g -= alpha_cg*AP;
		projector.PHI(g, phi);  assert_ge(g.dot(phi),0);
		precond.solve(g,z, phi);   assert_eq(z, z);
		assert_ge(g.dot(z),0);
		const T beta = (z.dot(AP)) / (p.dot(AP)); assert_eq(beta, beta);
		p = z-beta*p;
		residual_out = computeGradients(g, false);
		result_is_prop = proportional(result);

		A.multiply(p, AP);
		const T pd = p.dot(AP); assert_ge(pd, 0); // pd = p^t*A*p > 0
		alpha_cg = (z.dot(g))/pd; assert_ge(alpha_cg, 0);
		y = result-alpha_cg*p;
	    fx = fy;
	    fy = projectFunValue(y);

		num_cg ++;
		iteration ++;
	  }

	  return residual_out;
	}

  protected:
	//problem
	const MAT& A;
	const Vec& B;

	//parameter
	size_t max_iterations;
	T tolerance_factor;
	T gamma, alpha_bar;

	// internal structures
	PRECONDITION &precond;
	PROJECTOIN &projector;
	
	//temporary
	Vec g,p,z,beta,phi,gp,y;
	size_t iterations_out;
	T residual_out;

	// solving info
	int result_code;
	int num_cg, num_exp, numprop, iteration;
	T fun_val;
  };

  template <typename T, typename MAT, typename PROJECTOIN,typename PRECONDITION>
  class MPRGP:public MPRGPBase<T,MAT, PROJECTOIN, PRECONDITION>{

	typedef Eigen::Matrix<T,-1,1> Vec;
	typedef MPRGPBase<T,MAT,PROJECTOIN, PRECONDITION> MB;

  public:
	MPRGP(const MAT &A,const Vec &B,PRECONDITION &precond, 
		  PROJECTOIN &projector,const int max_it = 1000,const T tol=1e-3):
	  MB(A,B,precond, projector, max_it, tol){}

	int solve(Vec &result){

	  FUNC_TIMER_CLASS timer("mprgp solving");
	  this->initialize(result);

	  for( MB::iteration = 0; MB::iteration < MB::max_iterations; MB::iteration++ ){

	  	DEBUG_LOG("mprgp step "<<MB::iteration);
	  	MB::residual_out = this->computeGradients(MB::g);

	  	if( MB::residual_out <= MB::tolerance_factor ){
	  	  MB::iterations_out = MB::iteration;
	  	  MB::result_code = 0;
	  	  break;
	  	}

	  	if( MB::proportional(result) ){

	  	  Vec& AP=MB::gp;
	  	  MB::A.multiply(MB::p,AP);
	  	  const T pd = MB::p.dot(AP); assert_ge(pd, 0); // pd = p^t*A*p > 0
	  	  const T alpha_cg = (MB::z.dot(MB::g)) / pd; assert_ge(alpha_cg, 0);
	  	  const T alpha_f = MB::projector.stepLimit(result,MB::p,alpha_cg); 
		  assert_ge(alpha_f, 0);

	  	  if(alpha_cg <= alpha_f){
 	  		this->CGStep(AP, alpha_cg, result);
	  	  }else{
	  		this->ExpStep(AP, alpha_f, result);
	  	  }
	  	}else{
	  	  this->PropStep(result);
	  	}
	  }

	  ERROR_LOG_COND("MPRGP is not convergent with "<< MB::iteration << " iterations."<<endl, (MB::iteration < MB::max_iterations));
	  DEBUG_FUN( this->printSolveInfo() );
	  return MB::result_code;
	}

  };

  template <typename T, typename MAT, typename PROJECTOIN,typename PRECONDITION>
  class MPRGPMonotonic:public MPRGPBase<T,MAT, PROJECTOIN, PRECONDITION>{

  	typedef Eigen::Matrix<T,-1,1> Vec;
  	typedef MPRGPBase<T,MAT,PROJECTOIN, PRECONDITION> MB;

  public:
  	MPRGPMonotonic(const MAT &A,const Vec &B,PRECONDITION &precond, 
  				   PROJECTOIN &projector,const int max_it = 1000,const T tol=1e-3):
  	  MB(A,B,precond, projector, max_it, tol){}

  	int solve(Vec &result){

	  FUNC_TIMER_CLASS timer("monotonic mprgp solving");
  	  this->initialize(result);

  	  for( MB::iteration = 0; MB::iteration < MB::max_iterations; ){

  	  	MB::residual_out = this->computeGradients(MB::g);
  	  	if( MB::residual_out <= MB::tolerance_factor )
  	  	  break;

		bool result_is_prop = MB::proportional(result);
  	  	if(result_is_prop)
  		  MB::residual_out = this->CGMonotonicStep(result, result_is_prop);

  	  	if( MB::residual_out <= MB::tolerance_factor )
  	  	  break;

  		if ( !(isFeasible(MB::projector.getPlanes(), MB::y)) ){
  		  this->ExpMonotonicStep(result);
		  MB::iteration ++;
  	  	}else if( !result_is_prop ){
  		  MB::A.multiply(result, MB::g);
  		  MB::g -= MB::B;
  	  	  this->PropStep(result);
		  MB::iteration ++;
  	  	}
  	  }

	  MB::z = result;
	  MB::projector.project(MB::z, result);
	  
  	  MB::result_code = MB::residual_out <= MB::tolerance_factor ? 0 : -1;
  	  MB::iterations_out = MB::iteration;
  	  ERROR_LOG_COND("MPRGP is not convergent with "<< MB::iteration << " iterations."<<endl, (MB::iteration < MB::max_iterations));

	  DEBUG_FUN(MB::projector.DECIDE_FACE(result));
  	  DEBUG_FUN( this->printSolveInfo() );
  	  return MB::result_code;
  	}

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
	T funcValue(const VEC& x,const VEC_SECOND&b)const{
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

	const SparseMatrix<T> &getMatrix()const{
	  return A;
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
	template <typename MAT, typename Preconditioner=DiagonalPlanePreconSolver<T,MAT, false> >
	static int solve(const MAT &A,const Vec &B, PlaneProjector<T> &projector, Vec &x, const T tol=1e-3, const int max_it = 1000){

	  assert_eq(A.rows(),B.size());
	  assert_eq(A.rows(),x.size());
	  FUNC_TIMER_CLASS fun_timer("total solving");

	  UTILITY::Timer timer;
	  timer.start();
	  Preconditioner precond(A, projector.getFace(), projector.getPlanes());
	  timer.stop("time for preconditioning setup: ");

	  typedef MPRGPMonotonic<T, MAT, PlaneProjector<T>, Preconditioner > MPRGPSolver;
	  MPRGPSolver solver(A, B, precond, projector, max_it, tol);
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
	  convert<T>(planes, planes_for_each_node, x.size()/3);
	  return solve(A,B,planes_for_each_node, x, tol, max_it);
	}

	// load the problem from file, then solve it.
	static int solve(const string file_name, Vec&x, const T tol=1e-3, const int max_it = 1000){

	  SparseMatrix<T> A;
	  Vec B;
	  VVec4X planes;
	  int code = -1;
	  if (loadQP(A,B,planes,x,file_name)){
		code = solve(FixedSparseMatrix<double>(A),B,planes,x,tol,max_it);
	  }
	  return code;
	}

	// compute the lagragian multipliers, where g = Ax-B.
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

	// Compute the lagragian gradients. As input, we require lag_grad = Ax-b.
	static void getLagGrad(const VVVec4X &planes, const vector<vector<T> > &all_lambdas, Vec &lag_grad){

	  assert_eq( lag_grad.size(), (int)planes.size()*3 );
	  assert_eq( planes.size(), all_lambdas.size() );
	  for (size_t i = 0; i < all_lambdas.size(); ++i){
		const vector<T> &lambdas = all_lambdas[i];
		const VVec4X &plane = planes[i];
		assert_eq(lambdas.size(), plane.size());
		for (size_t p = 0; p < lambdas.size(); ++p)
		  lag_grad.template segment<3>(i*3) -= lambdas[p]*plane[p].template segment<3>(0);
	  }
	}

	template <typename MAT>
	static bool checkResult(const MAT &A,const Vec &B, const PlaneProjector<T> &projector, 
							const Vec &x,const T grad_tol, const T lambda_low_bound=0.0){

	  const Vec g = A*x-B;
	  const vector<vector<int> > &face_indices = projector.getFaceIndex();
	  const VVVec4X &planes_for_each_node = projector.getPlanes();
	  vector<vector<T> > all_lambdas;
	  computeLagMultipliers(g,planes_for_each_node,face_indices,all_lambdas);

	  bool valid = greaterThan(all_lambdas, lambda_low_bound);
	  Vec lag_grad = g;
	  getLagGrad(planes_for_each_node, all_lambdas, lag_grad);
	  const T gnorm = lag_grad.norm();
	  ERROR_LOG_COND("lag_grad.norm(): "<< gnorm << "\ntol: "<<grad_tol, (gnorm <= grad_tol));
	  valid &= (gnorm <= grad_tol);
	  return valid;
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
		// assert_ge(lambdas[p],0.0f);

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

		// assert_ge(lambdas[p0],0.0f);
		// assert_ge(lambdas[p1],0.0f);

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

		// assert_ge(lambdas[p0],0.0f);
		// assert_ge(lambdas[p1],0.0f);
		// assert_ge(lambdas[p2],0.0f);
	  }
	}

  };
  
}//end of namespace

#endif /* _MPRGPSOLVER_H_ */
