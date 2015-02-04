#ifndef _MPRGPPRECONDITION_H_
#define _MPRGPPRECONDITION_H_

#include <iostream>
#include <stdio.h>
#include <vector>
#include <set>
#include <eigen3/Eigen/Sparse>
#include <eigen3/Eigen/Dense>
#include <boost/shared_ptr.hpp>
#include <MPRGPUtility.h>
#include <SparseMatrixTools.h>
using namespace Eigen;
using namespace std;

namespace MATH{

  // preconditioners for boundary constraints
  // no preconditioning
  template <typename T, typename MAT >
  class InFaceNoPreconSolver{

	typedef Eigen::Matrix<T,-1,1> Vec;

  public:
	InFaceNoPreconSolver(const vector<char>& face):_face(face){}
	int solve(const Vec&rhs, Vec&result, const Vec &phi){
	  MASK_FACE(rhs,result,_face);
	  return 0;
	}
	void setMatrix(const MAT& matrix){}

  protected:
	const vector<char>& _face;
  };

  // Jacobian preconditioner
  template <typename T, typename MAT, bool NO_PRECOND=false >
  class DiagonalInFacePreconSolver{

	typedef Eigen::Matrix<T,-1,1> Vec;

  public:
	DiagonalInFacePreconSolver(const MAT &M,const vector<char> &face):face(face){

	  invert_diag.resize(M.rows());
	  if (!NO_PRECOND){
		for (int i = 0; i < invert_diag.size(); ++i){
		  assert_ge( M.diag(i), ScalarUtil<T>::scalar_eps );
		  invert_diag[i] = 1.0/M.diag(i);
		}
	  }
	}
	int solve(const Vec &g, Vec &z, const Vec &phi)const{

	  if (!NO_PRECOND){
		z.resize(phi.size());
		for(int i = 0; i < z.size(); i++){
		  assert_eq(invert_diag[i], invert_diag[i]);
		  z[i] = phi[i]*invert_diag[i];
		}
	  }else{
		z = phi;
	  }
	  return 0;
	}

  protected:
	Vec invert_diag;
	const vector<char> &face;
  };



  // preconditioners for decoupled constraints
  // Jacobian preconditioner
  template <typename T, typename MAT, bool NO_PRECOND=false >
  class DiagonalDecouplePreconSolver:public DiagonalInFacePreconSolver<T,MAT,NO_PRECOND>{

	typedef Eigen::Matrix<T,-1,1> Vec;

  public:
	DiagonalDecouplePreconSolver(const MAT &M,const vector<char> &face,const SparseMatrix<T> &J):
	  DiagonalInFacePreconSolver<T,MAT,NO_PRECOND>(M,face){

	  if (!NO_PRECOND && J.rows() > 0){
		JMinv = J*this->invert_diag.asDiagonal();
		const SparseMatrix<T> JMinvJt = JMinv*J.transpose();
		getDiagonal(JMinvJt, invert_JMinvJt);
		for (int i = 0; i < invert_JMinvJt.size(); ++i){
		  assert_ge( invert_JMinvJt[i], ScalarUtil<T>::scalar_eps );
		  invert_JMinvJt[i] = 1.0/invert_JMinvJt[i];
		}
	  }
	}
	int solve(const Vec &g, Vec &z, const Vec &phi)const{

	  if (!NO_PRECOND && JMinv.rows() > 0){
		Vec lambda = invert_JMinvJt.asDiagonal()*(JMinv*g);
		assert_eq(lambda.size(), this->face.size());
		for (size_t i = 0; i < this->face.size(); ++i){
		  if(0 == this->face[i])
			lambda[i] = 0;
		}
		z = this->invert_diag.asDiagonal()*g-JMinv.transpose()*lambda;
	  }else{
		z = phi;
	  }
	  return 0;
	}
	
  private:
	SparseMatrix<T> JMinv; // J*M^{-1}
	Vec invert_JMinvJt;
  };



  // preconditioners for plane constraints
  // Jacobian preconditioner
  template <typename T, typename MAT,bool NO_PRECOND=false>
  class DiagonalPlanePreconSolver{

	typedef Eigen::Matrix<T,-1,1> Vec;
	typedef Eigen::Matrix<T,3,1> Vec3X;
	typedef Eigen::Matrix<T,4,1> Vec4X;

  public:
	DiagonalPlanePreconSolver(const MAT &A,const vector<char>& face,
							  const VVVEC4X_T&planes,const vector<vector<int> >&face_indices):
	  A(A), face(face), planes(planes), face_indices(face_indices){

	  if (!NO_PRECOND){
		inv_diag.resize(A.rows());
		for (int i = 0; i < inv_diag.size(); ++i){
		  assert_ge( A.diag(i), ScalarUtil<T>::scalar_eps );
		  inv_diag[i] = 1.0/A.diag(i);
		}
	  }
	}

	void solve(const Vec&g, Vec&z, const Vec &phi)const{

	  if (NO_PRECOND){
		z = phi;
	  }else{
		project(g,z);
		for ( int i = 0; i < g.size(); i++){
		  z[i] = z[i]*inv_diag[i];
		}
	  }
	}

	void project(const Vec&g, Vec&z)const{

	  z = g;
	  const int n = g.size()/3;

	  for (int i = 0; i < n; ++i){

		const int j = i*3;

		if (1 == face[j]){

		  const int fi = face_indices[i][0];
		  const Vec3X n = planes[i][fi].template segment<3>(0);
		  const Vec3X temp = z.template segment<3>(j)-(z[j]*n[0]+z[j+1]*n[1]+z[j+2]*n[2])*n;
		  if(g.template segment<3>(j).dot(temp) < 0){

			z.template segment<3>(j).setZero();
		  }else{

			const T zm = z[j]*inv_diag[j]*n[0] + z[j+1]*inv_diag[j+1]*n[1] + z[j+2]*inv_diag[j+2]*n[2];
			const T sm = inv_diag[j]*n[0]*n[0] + inv_diag[j+1]*n[1]*n[1] + inv_diag[j+2]*n[2]*n[2];
			assert_gt(sm, ScalarUtil<T>::scalar_eps);
			z.template segment<3>(j) -= n*(zm/sm);
		  }
		}else if(2 == face[j]){

		  const int f0 = face_indices[i][0];
		  const int f1 = face_indices[i][1];
		  assert_ne(f0,f1);
		  const Vec4X &pi = planes[i][f0];
		  const Vec4X &pj = planes[i][f1];

		  Vec3X invL, N, L;
		  invL[0] = sqrt(inv_diag[j+0]);
		  invL[1] = sqrt(inv_diag[j+1]);
		  invL[2] = sqrt(inv_diag[j+2]);
		  
		  L[0] = 1.0/invL[0];
		  L[1] = 1.0/invL[1];
		  L[2] = 1.0/invL[2];

		  N[0] = invL[2]*invL[1]*(pi[1]*pj[2]-pi[2]*pj[1]);
		  N[1] = invL[0]*invL[2]*(pi[2]*pj[0]-pi[0]*pj[2]);
		  N[2] = invL[0]*invL[1]*(pi[0]*pj[1]-pi[1]*pj[0]);

		  const T LgN = invL[0]*z[j+0]*N[0] + invL[1]*z[j+1]*N[1] + invL[2]*z[j+2]*N[2];
		  z[j+0] = LgN*L[0]*N[0];
		  z[j+1] = LgN*L[1]*N[1];
		  z[j+2] = LgN*L[2]*N[2];

		}else if(face[j] >= 3){

		  z.template segment<3>(j).setZero();
		}
	  }
	}

	void projectMatrix(const SparseMatrix<T> &M, SparseMatrix<T> &D)const{

	  D.resize(M.rows(), M.cols());
	  D.setZero();

	  vector<Triplet<T> > triplet;
	  for (int k = 0; k < M.outerSize(); ++k){
		for(typename SparseMatrix<T>::InnerIterator it(M,k);it;++it){
		  const int r = it.row();
		  const int c = it.col();
		  if( (planes[r/3].size()==0 && planes[c/3].size()==0) || (r==c))
			triplet.push_back(Triplet<T>(r,c,it.value()));
		}
	  }
	  D.reserve(triplet.size());
	  D.setFromTriplets(triplet.begin(), triplet.end());
	}

  protected:
	const MAT &A;
	const vector<char>& face;
	const VVVEC4X_T &planes;
	const vector<vector<int> > &face_indices;
	Vec inv_diag;
  };

  // Jacobian preconditioner
  template <typename T, typename MAT >
  class BlockDiagonalPlanePreconSolver{

	typedef Eigen::Matrix<T,-1,1> Vec;
	typedef Eigen::Matrix<T,3,1> Vec3X;
	typedef Eigen::Matrix<T,3,3> Mat3X;
	typedef std::vector<Mat3X, Eigen::aligned_allocator<Mat3X> > VMat3X;

  public:
	BlockDiagonalPlanePreconSolver(const MAT&A,const vector<char>&face,
								   const VVVEC4X_T&P,const vector<vector<int> >&face_indices):
	  A(A), face(face), planes(P), face_indices(face_indices){

	  const SparseMatrix<T> &M = A.getMatrix();

	  D.resize(M.rows()/3);
	  for (int k = 0; k < M.outerSize(); ++k){
		for (typename SparseMatrix<T>::InnerIterator it(M,k); it; ++it){
		  const int r = it.row();
		  const int c = it.col();
		  if(r/3 == c/3){
			D[r/3](r%3,c%3) = it.value();
		  }
		}
	  }

	  inv_D.resize(D.size());
	  for (int i = 0; i < D.size(); ++i){
		inv_D[i] = D[i].inverse();
		assert_eq(inv_D[i], inv_D[i]);
	  }
	}

	void solve(const Vec&g, Vec&z, const Vec &phi)const{

	  project(g,z);
	  for ( int i = 0; i < g.size(); i+=3){
		const Vec3X zi = z.template segment<3>(i);
		z.template segment<3>(i) = inv_D[i/3]*zi;
	  }
	}

  protected:
	void project(const Vec&g, Vec&z)const{

	  z = g;
	  const int n = g.size()/3;
	  for (int i = 0; i < n; ++i){
		const int j = i*3;
		if (face[j] != 0){
		  assert_eq(face[j],1);
		  assert_eq(planes[i].size(), 1);
		  const Vec3X n = planes[i][0].template segment<3>(0);
		  const Vec3X temp = z.template segment<3>(j)-(z[j]*n[0]+z[j+1]*n[1]+z[j+2]*n[2])*n;
		  if(g.template segment<3>(j).dot(temp) < 0){
			z.template segment<3>(j).setZero();
		  }else{
			const Vec3X dn = inv_D[i]*n;
			const T zm = z.template segment<3>(j).dot(dn);
			const T sm = n.dot(dn);
			assert_gt(sm, ScalarUtil<T>::scalar_eps);
			z.template segment<3>(j) -= n*(zm/sm);
		  }
		}
	  }
	}

  protected:
	const MAT &A;
	const vector<char>& face;
	const VVVEC4X_T &planes;
	const vector<vector<int> >&face_indices;
	VMat3X D, inv_D;
  };

  // Cholesky preconditioner
  template <typename T, typename MAT >
  class CholeskyPlanePreconSolver:public DiagonalPlanePreconSolver<T,MAT>{

	typedef Eigen::Matrix<T,-1,1> Vec;

  public:
	CholeskyPlanePreconSolver(const MAT &A,const vector<char>&face,const VVVEC4X_T&P,const vector<vector<int> >&face_indices):
	  DiagonalPlanePreconSolver<T,MAT>(A,face,P,face_indices){

	  this->projectMatrix( A.getMatrix(), D );
	  chol.compute(D);
	  ERROR_LOG_COND("Factorization Fail!", chol.info()==Eigen::Success);
	}

	void solve(const Vec&g, Vec&z, const Vec &phi){

	  this->project(g,z);
	  z = chol.solve(z);
	}

  private:
	IncompleteLUT<T> chol;
	SparseMatrix<T> D;
  };

  // Symmetric Gaussian-Seidel preconditioner
  template <typename T, typename MAT >
  class SymGaussSeidelPlanePreconSolver:public DiagonalPlanePreconSolver<T,MAT>{

	typedef Eigen::Matrix<T,-1,1> Vec;

  public:
	SymGaussSeidelPlanePreconSolver(const MAT &A,const vector<char>&face,const VVVEC4X_T&P,const vector<vector<int> >&face_indices):
	  DiagonalPlanePreconSolver<T,MAT>(A,face,P,face_indices){

	  const SparseMatrix<T> &M = A.getMatrix();
	  SparseMatrix<T> L(M.rows(), M.cols());

	  vector<Triplet<T> > triplet;
	  for (int k = 0; k < M.outerSize(); ++k){
		for(typename SparseMatrix<T>::InnerIterator it(M,k);it;++it){
		  const int r = it.row();
		  const int c = it.col();
		  if( r >= c)
			if( (P[r/3].size()==0 && P[c/3].size()==0) || (r==c) )
			  triplet.push_back(Triplet<T>(r,c,it.value()));
		}
	  }

	  L.reserve(triplet.size());
	  L.setFromTriplets(triplet.begin(), triplet.end());
	  D = (L*this->inv_diag.asDiagonal())*(L.transpose());
	  
	  chol.compute(D);
	  ERROR_LOG_COND("Factorization Fail!", chol.info()==Eigen::Success);
	}

	void solve(const Vec&g, Vec&z, const Vec &phi){

	  this->project(g,z);
	  z = chol.solve(z);
	}

  private:
    SimplicialCholesky<SparseMatrix<T,0> > chol;
	SparseMatrix<T> D;
  };

  // Tridiagonal preconditioner
  template <typename T, typename MAT >
  class TridiagonalPlanePreconSolver:public DiagonalPlanePreconSolver<T,MAT>{

	typedef Eigen::Matrix<T,-1,1> Vec;

  public:
	TridiagonalPlanePreconSolver(const MAT &A,const vector<char>&face,const VVVEC4X_T&P,const vector<vector<int> >&face_indices):
	  DiagonalPlanePreconSolver<T,MAT>(A,face,P,face_indices){

	  SparseMatrix<T> D;
	  this->projectMatrix( A.getMatrix(), D );
	  buildTridiagonalMatrix(D, TriM);

	  chol.compute(TriM);
	  ERROR_LOG_COND("Factorization Fail!", chol.info()==Eigen::Success);
	}

	void solve(const Vec&g, Vec&z, const Vec &phi){

	  this->project(g,z);
	  z = chol.solve(z);
	}

	static void buildTridiagonalMatrix(const SparseMatrix<T> &M, SparseMatrix<T> &Tri){
	  
	  vector<Triplet<T> > triplet;
	  for (int k = 0; k < M.outerSize(); ++k){
		for(typename SparseMatrix<T>::InnerIterator it(M,k);it;++it){
		  const int r = it.row();
		  const int c = it.col();
		  if( (r+1==c) || (r==c) || (r-1==c) )
			triplet.push_back(Triplet<T>(r,c,it.value()));
		}
	  }
	  Tri.resize(M.rows(), M.cols());
	  Tri.setZero();
	  Tri.reserve(triplet.size());
	  Tri.setFromTriplets(triplet.begin(), triplet.end());
	}

  private:
    SimplicialCholesky<SparseMatrix<T,0> > chol;
	SparseMatrix<T> TriM;
  };

  // ADI preconditioner with one single direction.
  template <typename T, typename MAT >
  class SingleADIPlanePreconSolver:public DiagonalPlanePreconSolver<T,MAT>{

	typedef Eigen::Matrix<T,-1,1> Vec;
	typedef std::vector<std::set<int> > VVI;
	typedef boost::shared_ptr<SimplicialCholesky<SparseMatrix<T,0> > > pLLT;

  public:
	SingleADIPlanePreconSolver(const MAT&A,const vector<char>&f,const VVVEC4X_T&P,const VVI&g,const vector<vector<int> >&face_indices):
	  DiagonalPlanePreconSolver<T,MAT>(A,f,P,face_indices){

	  this->projectMatrix( A.getMatrix(), D );
	  buildSolvers(D, g);
	}

	void solve(const Vec&g, Vec&z, const Vec &phi){

	  this->project(g,temp_x);
	  solve_imp(temp_x, z, phi);
	}
	void solve_imp(const Vec&projected_x, Vec&z, const Vec &phi){

	  z.resize(projected_x.size());
	  z.setZero();
	  for (size_t i = 0; i < S.size(); ++i){
		z += S[i].transpose()*(sol[i]->solve(S[i]*projected_x));
	  }
	}
	
  protected:
	void buildSolvers(const SparseMatrix<T> &D, const VVI&groups){

	  S.resize(groups.size());
	  TriMat.resize(groups.size());
	  sol.resize(groups.size());
	  for (size_t i = 0; i < groups.size(); ++i){
		EIGEN3EXT::genReshapeMatrix(D.rows(), 3, groups[i], S[i], false);
		TridiagonalPlanePreconSolver<T,MAT>::buildTridiagonalMatrix(S[i]*D*S[i].transpose(), TriMat[i]);
		sol[i] = pLLT(new SimplicialCholesky<SparseMatrix<T,0> >());
		sol[i]->compute(TriMat[i]);
		ERROR_LOG_COND("Factorization Fail!", sol[i]->info()==Eigen::Success);
	  }
	}

  private:
    vector<pLLT > sol;
	vector<SparseMatrix<T> > S;// reshape matrices
	vector<SparseMatrix<T> > TriMat;
	SparseMatrix<T> D;
	Vec temp_x;
  };

  // ADI preconditioner
  template <typename T, typename MAT >
  class ADIPlanePreconSolver{

	typedef Eigen::Matrix<T,-1,1> Vec;
	typedef std::vector<std::vector<std::set<int> > > VVVI;
	typedef boost::shared_ptr<SingleADIPlanePreconSolver<T, MAT> > pSingleADI;

  public:
	ADIPlanePreconSolver(const MAT&A,const vector<char>&f,const VVVEC4X_T&P,const VVVI&g,const vector<vector<int> >&face_indices){

	  direction = 0;
	  assert_eq(g.size(), 3);
	  for (int i = 0; i < 3; ++i){
		adi[i] = pSingleADI(new SingleADIPlanePreconSolver<T,MAT>(A,f,P,g[i],face_indices));
	  }
	}

	void solve(const Vec&g, Vec&z, const Vec &phi){

	  // adi[0]->project(g,temp_x);

	  // adi[0]->solve_imp(temp_x, z);
	  // adi[1]->solve_imp(z, temp_x);
	  // adi[2]->solve_imp(temp_x, z);
	  
	  // adi[2]->solve_imp(z, temp_x);
	  // adi[1]->solve_imp(temp_x, z);
	  // adi[0]->solve_imp(z, temp_x);

	  // z = temp_x;

	  adi[direction]->solve(g, z, phi);
	  direction = (direction+1)%3;
	}
	
  private:
	int direction;
	pSingleADI adi[3];
	// Vec temp_x;
  };

}//end of namespace

#endif /* _MPRGPPRECONDITION_H_ */
