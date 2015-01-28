#ifndef _PROJECTEDGS_H_
#define _PROJECTEDGS_H_

#include <eigen3/Eigen/Dense>
#include <eigen3/Eigen/Sparse>
using namespace Eigen;

namespace MATH{
  
  // implementation of Gauss Seidel method
  // for solving B lambda = c, s.t. lambda >= 0
  class ProjectedGaussSeidel{
	
  public:
	ProjectedGaussSeidel(const int max_it = 1000, const double tol = 1e-4):
	  max_it(max_it),tol(tol){}
	void reset(const SparseMatrix<double> &B);
	bool solve(const VectorXd &c, VectorXd &lambda);
	
  protected:
	const int max_it;
	const double tol;	
  };

  // implemented the ICA (Iterative Constraints Anticpation) solver of the paper:
  // Implicit Contact Handling for Deformable Objects, EG 2009, 
  // Miguel A. Otaduy, Rasmus Tamstorf, Denis Steinemann, and Markus Gross.
  // for sovling: 
  // A x = J^t lambda, s.t. 0 <= lambda _|_ Jx >= p.
  // In eq (6) of paper, we have: x = \Delta v, and p = -1/(\delta t)g0 - Jv*
  class ICASolver{
	
  public:
	ICASolver(const int max_it = 1000, const double tol = 1e-4):
	  max_it(max_it), tol(tol){}
	void reset(const SparseMatrix<double> &A);
	bool solve(const SparseMatrix<double> &J, const VectorXd &p, VectorXd &x){

	  B = J*(invDa*J.transpose());
	  PGS.reset(B);
	  
	  VectorXd lambda(J.rows()), c, rhs, pre_x;
	  lambda.setZero();

	  bool succ = false;
	  for (int it = 0; it < max_it; ++it){
		c = p-J*(invDa*(LaUa*x));
		PGS.solve(c, lambda);
		rhs = Ua*x+J.transpose()*lambda;
		pre_x = x;
		x = DaLa.solve(rhs);
		if((pre_x-x).norm() < tol){
		  succ = true;
		  break;
		}
	  }
	  return succ;
	}
	
  protected:
	const int max_it;
	const double tol;
	SparseMatrix<double> invDa;
	SparseMatrix<double> B;
	SparseMatrix<double> LaUa;
	ProjectedGaussSeidel PGS;
	SparseMatrix<double> Ua;
	SparseMatrix<double> DaLa;
  };
  
}//end of namespace

#endif /*_PROJECTEDGS_H_*/
