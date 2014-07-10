#ifndef _MPRGPPRECONDITION_H_
#define _MPRGPPRECONDITION_H_

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
	  for(size_t i=0; i<rhs.size(); i++){
		assert_ne( _matrix.diag(i), 0);
		// if( 0 == _face[i] )
		// result[i]=rhs[i]/_matrix.diag(i);
		result[i]=rhs[i];
		// else
		//   result[i]=0.0f;
	  }
	  return 0;
	}

  private:
	const MAT &_matrix;
	const vector<char>& _face;
  };

}//end of namespace


#endif /* _MPRGPPRECONDITION_H_ */
