#ifndef _MPRGPPRECONDITION_H_
#define _MPRGPPRECONDITION_H_

#include <iostream>
#include <stdio.h>
#include <vector>
#include <eigen3/Eigen/Sparse>
#include <eigen3/Eigen/Dense>
#include <boost/shared_ptr.hpp>
#include <MPRGPUtility.h>
using namespace Eigen;
using namespace std;

namespace MATH{

  // no preconditioning
  template <typename T, typename MAT >
  class InFaceNoPreconSolver{

	typedef Eigen::Matrix<T,-1,1> Vec;

  public:
	InFaceNoPreconSolver(const vector<char>& face):_face(face){}
	int solve(const Vec&rhs, Vec&result){
	  MASK_FACE(rhs,result,_face);
	  return 0;
	}
	void setMatrix(const MAT& matrix){}

  protected:
	const vector<char>& _face;
  };

  // Jacobian preconditioner
  template <typename T, typename MAT >
  class DiagonalInFacePreconSolver{

	typedef Eigen::Matrix<T,-1,1> Vec;

  public:
	DiagonalInFacePreconSolver(const MAT &M,const vector<char>& face):_matrix(M),_face(face){}
	int solve(const Vec&rhs, Vec&result){

	  result.resize(rhs.size());
	  assert_eq(_face.size(), rhs.size());
	  result = rhs;
	  for(size_t i=0; i<rhs.size(); i++){
	  	assert_ge( _matrix.diag(i), ScalarUtil<T>::scalar_eps );
	  	// if( 0 == _face[i] )
		//   result[i]=rhs[i]/_matrix.diag(i);
	  	// else
	  	//   result[i]=0.0f;
	  }
	  return 0;
	}

  private:
	const MAT &_matrix;
	const vector<char>& _face;
  };

  // Jacobian preconditioner
  template <typename T, typename MAT,bool NO_PRECOND=false>
  class DiagonalPlanePreconSolver{

	typedef Eigen::Matrix<T,-1,1> Vec;
	typedef Eigen::Matrix<T,3,1> Vec3X;

  public:
	DiagonalPlanePreconSolver(const MAT &A,const vector<char>& face,const VVVEC4X_T&planes):
	  A(A), face(face), planes(planes){

	  if (!NO_PRECOND){
		inv_diag.resize(A.rows());
		for (int i = 0; i < inv_diag.size(); ++i){
		  assert_ge( A.diag(i), ScalarUtil<T>::scalar_eps );
		  inv_diag[i] = 1.0/A.diag(i);
		}
	  }
	}

	void solve(const Vec&g, Vec&z)const{

	  if (NO_PRECOND){
		phi(g,z);
	  }else{
		project(g,z);
		for ( int i = 0; i < g.size(); i++){
		  z[i] = z[i]*inv_diag[i];
		}
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
		  const Vec3X temp = z.template segment<3>(j) - (z[j]*n[0] + z[j+1]*n[1] + z[j+2]*n[2])*n;
		  if(g.template segment<3>(j).dot(temp) < 0){
			z.template segment<3>(j).setZero();
		  }else{
			const T zm = z[j]*inv_diag[j]*n[0] + z[j+1]*inv_diag[j+1]*n[1] + z[j+2]*inv_diag[j+2]*n[2];
			const T sm = inv_diag[j]*n[0]*n[0] + inv_diag[j+1]*n[1]*n[1] + inv_diag[j+2]*n[2]*n[2];
			assert_gt(sm, ScalarUtil<T>::scalar_eps);
			z.template segment<3>(j) -= n*(zm/sm);
		  }
		}
	  }
	}

	void phi(const Vec&g, Vec&z)const{

	  z = g;
	  const int n = g.size()/3;
	  for (int i = 0; i < n; ++i){
		const int j = i*3;
		if (face[j] != 0){
		  assert_eq(face[j],1);
		  assert_eq(planes[i].size(), 1);
		  const Vec3X n = planes[i][0].template segment<3>(0);
		  const Vec3X temp = z.template segment<3>(j) - (z[j]*n[0] + z[j+1]*n[1] + z[j+2]*n[2])*n;
		  if(g.template segment<3>(j).dot(temp) < 0){
			z.template segment<3>(j).setZero();
		  }else{
			z.template segment<3>(j) = temp;
		  }
		}
	  }
	}

  protected:
	const MAT &A;
	const vector<char>& face;
	const VVVEC4X_T &planes;
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
	BlockDiagonalPlanePreconSolver(const MAT&A,const vector<char>&face,const VVVEC4X_T&P):
	  A(A), face(face), planes(P){

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

	void solve(const Vec&g, Vec&z)const{

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
	VMat3X D, inv_D;
  };

  // Cholesky preconditioner
  template <typename T, typename MAT >
  class CholeskyPlanePreconSolver:public DiagonalPlanePreconSolver<T,MAT>{

	typedef Eigen::Matrix<T,-1,1> Vec;
	typedef Eigen::Matrix<T,4,1> Vec4X;

  public:
	CholeskyPlanePreconSolver(const MAT &A,const vector<char>&face,const VVVEC4X_T&P):
	  DiagonalPlanePreconSolver<T,MAT>(A,face,P){

	  const SparseMatrix<T> &M = A.getMatrix();
	  D.resize(M.rows(), M.cols());
	  D.setZero();

	  vector<Triplet<T> > triplet;
	  for (int k = 0; k < M.outerSize(); ++k){
		for(typename SparseMatrix<T>::InnerIterator it(M,k);it;++it){
		  const int r = it.row();
		  const int c = it.col();
		  if( (P[r/3].size()==0 && P[c/3].size()==0) || (r==c))
			triplet.push_back(Triplet<T>(r,c,it.value()));
		}
	  }
	  D.reserve(triplet.size());
	  D.setFromTriplets(triplet.begin(), triplet.end());

	  chol.compute(D);
	  ERROR_LOG_COND("Factorization Fail!", chol.info()==Eigen::Success);
	}

	void solve(const Vec&g, Vec&z){

	  this->project(g,z);
	  z = chol.solve(z);
	}

  private:
	IncompleteLUT<T> chol;
	SparseMatrix<T> D;
  };

  // Symmetric Gauss-Seidel preconditioner
  template <typename T, typename MAT >
  class SymGaussSeidelPlanePreconSolver:public DiagonalPlanePreconSolver<T,MAT>{

	typedef Eigen::Matrix<T,-1,1> Vec;
	typedef Eigen::Matrix<T,4,1> Vec4X;

  public:
	SymGaussSeidelPlanePreconSolver(const MAT &A,const vector<char>&face,const VVVEC4X_T&P):
	  DiagonalPlanePreconSolver<T,MAT>(A,face,P){

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

	void solve(const Vec&g, Vec&z){

	  this->project(g,z);
	  z = chol.solve(z);
	}

  private:
    SimplicialCholesky<SparseMatrix<T,0> > chol;
	SparseMatrix<T> D;
  };

}//end of namespace

#endif /* _MPRGPPRECONDITION_H_ */
