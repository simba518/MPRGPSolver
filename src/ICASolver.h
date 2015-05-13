#ifndef _ICASOLVER_H_
#define _ICASOLVER_H_

#include <math.h>
#include <vector>
#include <string>
#include <eigen3/Eigen/Dense>
#include <eigen3/Eigen/Sparse>
#include "MPRGPUtility.h"
using namespace std;
using namespace Eigen;

namespace MATH{
  
  // implementation of Gauss Seidel method
  // for solving B lambda = c,
  // if project_lambda is true, then we require lambda >= 0.
  // see http://image.diku.dk/kenny/download/erleben.13.siggraph.course.notes.pdf, 
  // page 18, method PSOR.
  template<bool projected = false>
  class GaussSeidel{
	
  public:
	GaussSeidel(const int max_it = 1000, const double tol = 1e-4):
	  max_it(max_it),tol(tol), iterations(-1), residual(-1.0f),pB(NULL),name("GaussSeidel"){}

	void reset(const SparseMatrix<double> &B){

	  //decompose B as: B = Lb+Db+Ub
	  assert_eq(B.rows(), B.cols());
	  const int nz = B.nonZeros();
	  const int r = B.rows();

	  pB = &B;
	  
	  Li.resize(r);
	  Lb.resize(r);
	  invD.resize(r);
	  for (int i = 0; i < r; ++i){
		Li[i].clear();
		Li[i].reserve( min(i,nz/r) );
		Lb[i].clear();
		Lb[i].reserve( min(i,nz/r) );
	  }
	  
	  vector<Triplet<double> > Ub_triplet;
	  for (int k = 0; k < B.outerSize(); ++k){
		for (SparseMatrix<double>::InnerIterator it(B,k); it; ++it){
		  const int r = it.row();
		  const int c = it.col();
		  if (r == c){
			invD[r] = 1.0f/it.value();
			assert_eq_ext(invD[r], invD[r], "(r,c,v): "<<r<<", "<<c<<", "<<it.value());
		  }else if (r > c){
			Li[r].push_back(c);
			Lb[r].push_back(it.value());
		  }else{
			Ub_triplet.push_back( Triplet<double>(r,c,it.value()) );
		  }
		}
	  }

	  Ub.resize(r,r);
	  Ub.reserve( Ub_triplet.size() );
	  Ub.setFromTriplets( Ub_triplet.begin(), Ub_triplet.end() );
	}

	bool solve(const VectorXd &c, VectorXd &lambda){

	  {// check dimensions
		assert_eq(lambda.size(), c.size());
		assert_eq(lambda.size(), (int)invD.size());
		assert_eq(lambda.size(), Ub.rows());
		assert_eq(lambda.size(), (int)Lb.size());
		assert_eq(lambda.size(), (int)Li.size());
	  }

	  bool succ = false;
	  for (iterations = 0; iterations < max_it; ++iterations){
		
		if (projected)
		  pre_lambda = lambda;
		rhs = c - Ub*lambda;

		for (int row = 0; row < (int)invD.size(); ++row){
		  for (int i = 0; i < (int)Li[row].size(); ++i){
			const int col = Li[row][i];
			rhs[row] -= lambda[col]*Lb[row][i];
		  }
		  lambda[row] = rhs[row]*invD[row];
		  if (projected) // Projection of GS
			lambda[row] = lambda[row] > 0 ? lambda[row]:0;
		}

		if (projected){
		  residual = (lambda-pre_lambda).norm();
		}else{
		  residual = ((*pB)*lambda-c).norm();
		}

		DEBUG_LOG(name << " residual = " << residual);
		if( residual <= tol ){
		  succ = true;
		  break;
		}
	  }
	  return succ;
	}

	double getResidual()const{
	  return residual;
	}

	int getIterations()const{
	  return iterations;
	}

	void setName(const string name){
	  this->name = name;
	}

	void printSolveInfo(const SparseMatrix<double> &B, const VectorXd &c, const VectorXd &lambda)const{

	  INFO_LOG(name << " residual: " << getResidual());
	  INFO_LOG(name << " iterations: " << getIterations());
	  if (projected && (pre_lambda.size() == lambda.size()) ){
		INFO_LOG(name << " Eqation residual ||pre_lambda - lambda||: " << (lambda - pre_lambda).norm());
	  }else{
		INFO_LOG(name << " Eqation residual ||B lambda - c||: " << (B*lambda - c).norm());
	  }
	  INFO_LOG(name << " Complement Cond lambda.dot(B lambda - c): " << lambda.dot((*pB)*lambda-c));
	  INFO_LOG(name << " lambda: " << lambda.transpose());
	  INFO_LOG(name << " B lambda - c: " << (B*lambda-c).transpose());
	}
	
  protected:
	const int max_it;
	const double tol;	

	SparseMatrix<double> Ub;
	vector<vector<int> > Li;
	vector<vector<double> > Lb;
	vector<double> invD;
	VectorXd rhs, pre_lambda;

	int iterations;
	double residual;
	const SparseMatrix<double> *pB;
	string name;
  };
  typedef GaussSeidel<true> ProjectedGaussSeidel;

  // 3x3 block GS
  class BlockGaussSeidel{
	
  public:
	BlockGaussSeidel(const int max_it = 1000, const double tol = 1e-4):
	  max_it(max_it), tol(tol), iterations(-1), residual(-1.0f), pA(NULL), name("BlockGaussSeidel"){}

	void reset(const SparseMatrix<double> &A){
	  
	  // decompose A as: A = La+Da+Ua
	  // @note in the original paper, we have A = -La+Da-Ua.
	  pA = &A;
	  assert_eq(A.rows(), A.cols());
	  assert_eq(A.rows()%3,0);
	  const int nz = A.nonZeros();
	  const int r = A.rows();

	  Ua.resize(r,r);
	  invDa.resize(r/3);

	  Li.resize(r);
	  La.resize(r);
	  for (int i = 0; i < r; ++i){
		Li[i].clear();
		Li[i].reserve(min(i,nz/r));
		La[i].clear();
		La[i].reserve(min(i,nz/r));
	  }

	  vector<Triplet<double> > Ua_triplet;
	  for (int k = 0; k < A.outerSize(); ++k){
		for (SparseMatrix<double>::InnerIterator it(A,k); it; ++it){
		  const int r = it.row();
		  const int c = it.col();
		  if ( r/3 == c/3 ){
			invDa[r/3](r%3,c%3) = it.value();
		  }else{
			if (r > c){
			  Li[r].push_back(c);
			  La[r].push_back(it.value());
			}else{
			  Ua_triplet.push_back( Triplet<double>(r,c,it.value()) );
			}
		  }
		}
	  }

	  Ua.reserve( Ua_triplet.size() );
	  Ua.setFromTriplets( Ua_triplet.begin(), Ua_triplet.end() );
	  
	  for (int i = 0; i < (int)invDa.size(); ++i){
		invDa[i] = invDa[i].inverse().eval();
		assert_eq(invDa[i], invDa[i]);
	  }
	}

	bool solve(const VectorXd &B, VectorXd &x){

	  {// check dimensions
		assert_eq(x.size(), Ua.rows());
		assert_eq(x.size(), (int)invDa.size()*3);
		assert_eq(x.size(), (int)Li.size());
		assert_eq(x.size(), (int)La.size());
	  }

	  bool succ = false;
	  VectorXd rhs(B.size());

	  residual = ((*pA)*x-B).norm();
	  DEBUG_LOG(name << " residual = " << residual);

	  for (iterations = 0; iterations < max_it; ++iterations){
		
		rhs = B-Ua*x;
		for (int row = 0; row < x.size(); row+=3){
		  for (int r = row; r < row+3; ++r){
			for (int j = 0; j < (int)Li[r].size(); ++j){
			  const int col = Li[r][j];
			  rhs[r] -= La[r][j]*x[col];
			}
		  }
		  x.segment<3>(row) = invDa[row/3]*rhs.segment<3>(row);
		}

		residual = ((*pA)*x-B).norm();
		DEBUG_LOG (name << " residual = " << residual);
		debug_fun ({const double func = 0.5*(x.dot((*pA)*x))-x.dot(B); DEBUG_LOG(name<<" func = "<<func);});
		if ( residual <= tol ) {
		  succ = true;
		  break;
		}
	  }
	  return succ;
	}

	double getResidual()const{
	  return residual;
	}

	int getIterations()const{
	  return iterations;
	}

	void setName(const string name){
	  this->name = name;
	}

	void printSolveInfo(const SparseMatrix<double>&A,const VectorXd&B,const VectorXd&x)const{
	  INFO_LOG(name << " residual: " << getResidual());
	  INFO_LOG(name << " iterations: " << getIterations());
	}
	
  protected:
	const int max_it;
	const double tol;

	SparseMatrix<double> Ua;
	vector<Matrix3d> invDa;
	vector<vector<int> > Li;
	vector<vector<double> > La;

	int iterations;
	double residual;

	const SparseMatrix<double> *pA;
	string name;
  };

  // implemented the ICA (Iterative Constraints Anticpation) solver of the paper:
  // Implicit Contact Handling for Deformable Objects, EG 2009, 
  // Miguel A. Otaduy, Rasmus Tamstorf, Denis Steinemann, and Markus Gross.
  // for sovling: 
  // A x = r + J^t lambda, s.t. 0 <= lambda _|_ Jx >= p.
  // In eq (6) of paper, we have: r=Av*-b, x = \Delta v, and p = -1/(\delta t)g0 - Jv*
  class ICASolver{
	
  public:
	ICASolver(const int max_it = 1000, const double tol = 1e-4, const int gs_max_it = 20, const double gs_tol = 1e-3):
	  max_it(max_it), tol(tol), iterations(-1), residual(-1.0f), pA(NULL), name("ICASolver"){
	  PGS = boost::shared_ptr<ProjectedGaussSeidel>(new ProjectedGaussSeidel(gs_max_it, gs_tol));
	}

	void reset(const SparseMatrix<double> &A, const VectorXd &r){
	  reset(A);
	  assert_eq(r.size(), A.rows());
	  R = r;
	}

	void reset(const SparseMatrix<double> &A){
	  
	  // decompose A as: A = La+Da+Ua
	  // @note in the original paper, we have A = -La+Da-Ua.
	  pA = &A;
	  R.resize(A.rows());
	  R.setZero();
	  assert_eq(A.rows(), A.cols());
	  assert_eq(A.rows()%3,0);
	  const int nz = A.nonZeros();
	  const int r = A.rows();

	  LaUa.resize(r,r);
	  Ua.resize(r,r);
	  invDa_Mat.resize(r,r);
	  invDa.resize(r/3);

	  Li.resize(r);
	  La.resize(r);
	  for (int i = 0; i < r; ++i){
		Li[i].clear();
		Li[i].reserve(min(i,nz/r));
		La[i].clear();
		La[i].reserve(min(i,nz/r));
	  }

	  vector<Triplet<double> > LaUa_triplet, Ua_triplet;
	  for (int k = 0; k < A.outerSize(); ++k){
		for (SparseMatrix<double>::InnerIterator it(A,k); it; ++it){
		  const int r = it.row();
		  const int c = it.col();
		  if ( r/3 == c/3 ){
			invDa[r/3](r%3,c%3) = it.value();
		  }else{
			LaUa_triplet.push_back( Triplet<double>(r,c,it.value()) );
			if (r > c){
			  Li[r].push_back(c);
			  La[r].push_back(it.value());
			}else{
			  Ua_triplet.push_back( Triplet<double>(r,c,it.value()) );
			}
		  }
		}
	  }

	  Ua.reserve( Ua_triplet.size() );
	  Ua.setFromTriplets( Ua_triplet.begin(), Ua_triplet.end() );
	  LaUa.reserve( LaUa_triplet.size() );
	  LaUa.setFromTriplets( LaUa_triplet.begin(), LaUa_triplet.end() );
	  
	  vector<Triplet<double> > invDa_triplet;
	  for (int i = 0; i < (int)invDa.size(); ++i){

		invDa[i] = invDa[i].inverse().eval();
		assert_eq(invDa[i], invDa[i]);
		const int n0 = i*3;
		for (int r = 0; r < 3; r++){
		  for (int c = 0; c < 3; c++)
			invDa_triplet.push_back( Triplet<double>(n0+r,n0+c,invDa[i](r,c)) );
		}
	  }
	  
	  invDa_Mat.reserve( invDa_triplet.size() );
	  invDa_Mat.setFromTriplets( invDa_triplet.begin(), invDa_triplet.end() );
	}

	bool solve(const SparseMatrix<double> &J, const VectorXd &p, VectorXd &x){

	  {// check dimensions
		assert_eq(J.cols(), x.size());
		assert_eq(J.rows(), p.size());
		assert_eq(x.size(), LaUa.rows());
		assert_eq(x.size(), Ua.rows());
		assert_eq(x.size(), invDa_Mat.rows());
		assert_eq(x.size(), (int)invDa.size()*3);
		assert_eq(x.size(), (int)Li.size());
		assert_eq(x.size(), (int)La.size());
	  }

	  B = J*(invDa_Mat*J.transpose());
	  PGS->reset(B);

	  lambda.resize(J.rows());
	  lambda.setZero();

	  bool succ = false;
	  for (iterations = 0; iterations < max_it; ++iterations){

		c = p+J*(invDa_Mat*(LaUa*x-R));
		PGS->solve(c, lambda);
		const VectorXd JtL = J.transpose()*lambda;
		rhs = JtL-Ua*x+R;

		// 3x3 block GS
		for (int row = 0; row < x.size(); row+=3){
		  for (int r = row; r < row+3; ++r){
			for (int j = 0; j < (int)Li[r].size(); ++j){
			  const int col = Li[r][j];
			  rhs[r] -= La[r][j]*x[col];
			}
		  }
		  x.segment<3>(row) = invDa[row/3]*rhs.segment<3>(row);
		}

		residual = ((*pA)*x-R-J.transpose()*lambda).norm();
		DEBUG_LOG(name << " residual = " << residual);
		debug_fun ({const double func = 0.5*(x.dot((*pA)*x))-x.dot(R); DEBUG_LOG(name<<" func = "<<func);});
		if ( residual <= tol ) {
		  succ = true;
		  break;
		}
	  }
	  return succ;
	}

	double getResidual()const{
	  return residual;
	}

	int getIterations()const{
	  return iterations;
	}

	const VectorXd &getLambda()const{
	  return lambda;
	}

	void setName(const string name){
	  this->name = name;
	}

	void printSolveInfo(const SparseMatrix<double> &A, const SparseMatrix<double> &J, 
						const VectorXd &p, const VectorXd &x)const{

	  INFO_LOG(name << " residual: " << getResidual());
	  INFO_LOG(name << " iterations: " << getIterations());
	  INFO_LOG(name << " Eqation residual ||A x - J^t lambda||: " << (A*x - J.transpose()*getLambda()).norm());
	  INFO_LOG(name << " Complement Cond lambda.dot(J x - p): " << getLambda().dot(J*x-p));
	  INFO_LOG(name << " lambda: " << getLambda().transpose());
	  INFO_LOG(name << " J x - p: " << (J*x-p).transpose());
	}
	
  protected:
	const int max_it;
	const double tol;

	SparseMatrix<double> LaUa;
	SparseMatrix<double> Ua;
	SparseMatrix<double> invDa_Mat;
	vector<Matrix3d> invDa;
	vector<vector<int> > Li;
	vector<vector<double> > La;

	SparseMatrix<double> B;

	VectorXd lambda, c, rhs, R;
	int iterations;
	double residual;

	const SparseMatrix<double> *pA;
	string name;
	boost::shared_ptr<ProjectedGaussSeidel> PGS;
  };
  
}//end of namespace

#endif /* _ICASOLVER_H_ */
