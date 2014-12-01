#ifndef _MPRGPUTILITY_H_
#define _MPRGPUTILITY_H_

#if defined(UTILITY_ASSERT)
#include <assertext.h>
#else/* no Utility/assertext.h  */
# define assert_ext(cond, info)
# define assert_eq(value_a,value_b)
# define assert_ne(value_a,value_b)
# define assert_ge(value_a,value_b)
# define assert_gt(value_a,value_b)
# define assert_le(value_a,value_b)
# define assert_lt(value_a,value_b)
# define assert_in(value_a,min,max)
# define assert_eq_ext(value_a,value_b,info)
# define assert_ne_ext(value_a,value_b,info)		
# define assert_ge_ext(value_a,value_b,info)		
# define assert_gt_ext(value_a,value_b,info)		
# define assert_le_ext(value_a,value_b,info)		
# define assert_lt_ext(value_a,value_b,info)		
# define assert_in_ext(value_a,min,max,info)
#endif /* UTILITY_ASSERT  */

#if defined(UTILITY_LOG)
#include <Log.h>
#else/* no Utility/assertext.h  */
#define PRINT_MSG_MICRO(title,event,cond)
#define PRINT_MSG_MICRO_EXT(title,event,cond,file,line)
#define ERROR_LOG_COND(event,cond)
#define ERROR_LOG(event)
#define WARN_LOG_COND(event,cond)
#define WARN_LOG(event)
#define TRACE_FUN()
#define INFO_LOG(event)
#define INFO_LOG_COND(event,cond)
#define DEBUG_LOG_EXT(event)
#define DEBUG_LOG(event)
#define CHECK_DIR_EXIST(dir)
#define CHECK_FILE_EXIST(f)
#endif /* UTILITY_LOG */

#if defined(UTILITY_TIMER)
#include <Timer.h>
#else/* no Utility/Timer.h */
#define FUNC_TIMER()
#endif /* UTILITY_LOG */

#include <iostream>
#include <fstream>
#include <omp.h>
#include <iomanip>
using namespace std;

#ifdef _MSC_VER
#define STRINGIFY(X) X
#define PRAGMA __pragma
#else
#define STRINGIFY(X) #X
#define PRAGMA _Pragma
#endif

// #define OMP_PARALLEL_FOR_ PRAGMA(STRINGIFY(omp parallel for num_threads(OmpSettings::getOmpSettings().nrThreads()) schedule(dynamic,OmpSettings::getOmpSettings().szChunk())))
#define OMP_PARALLEL_FOR_
#define OMP_CRITICAL_ PRAGMA(STRINGIFY(omp critical))

namespace MATH{

  template <typename T>
  struct ScalarUtil;
  template <>
  struct ScalarUtil<float> {
	static float scalar_max;
	static float scalar_eps;
  };
  template <>
  struct ScalarUtil<double> {
	static double scalar_max;
	static double scalar_eps;
  };

  template<typename VECTOR>
  inline void MASK_FACE(const VECTOR& in,VECTOR& out,const std::vector<char>& face){

	out.resize(in.size());
	assert(in.size() == face.size());
	OMP_PARALLEL_FOR_
	  for(size_t i=0;i<in.size();i++)
		if( 0 != face[i])
		  out[i]=0.0f;
		else 
		  out[i]=in[i];
  }

  template<typename VECTOR>
  inline int COUNT_CONSTRAINTS(const VECTOR& face){
	int c = 0;
	for (int i = 0; i < face.size(); ++i){
	  assert_ge((int)face[i],0);
	  c += (int)face[i];
	}
	return c;
  }

#define VVEC4X_T vector<Eigen::Matrix<T,4,1>, Eigen::aligned_allocator<Eigen::Matrix<T,4,1> > >

  // save the problem to the file: A, B, x0 and the constraints, i.e planes.
  // the problem is: 
  // min_{x} 1/2*x^t*A*x-x^t*B s.t. n_i*x_j+p_i>= 0
  template<typename T>
  inline bool writeQP(const Eigen::SparseMatrix<T> &A,const Eigen::Matrix<T,-1,1> &B,
					  const VVEC4X_T &planes,
					  const Eigen::Matrix<T,-1,1> &x0,const string file_name){
	  
	ofstream out;
	out.open(file_name.c_str());
	if (!out.is_open()){
	  ERROR_LOG("failed to open the file: "<<file_name);
	  return false;
	}

	// write A
	out << "dimension " << A.rows() << "\n";
	out << "planes "<< planes.size() << "\n";
	out << "A\n";
	out << "non_zeros "<< A.nonZeros() << "\n";
	for(int k=0;k<A.outerSize();++k)
	  for(typename Eigen::SparseMatrix<T>::InnerIterator it(A,k);it;++it)
		out << it.row()<<"\t"<<it.col()<<"\t"<<setprecision(12)<<it.value()<<"\n";
	  
	// write B
	out << "B\n";
	if (B.size() > 0) 
	  out<< setprecision(12) << B[0];
	for (int i = 1; i < B.size(); ++i)
	  out<< setprecision(12) << "\t" << B[i];
	out << "\n";

	// write P
	out << "P\n";
	for (int i = 0; i < planes.size(); ++i)
	  out<< setprecision(12) << planes[i][0] << "\t"<< planes[i][1] << "\t" << planes[i][2] << "\t"<< planes[i][3] << "\n";

	// write x0
	out << "x0\n";
	if (x0.size() > 0)
	  out<< setprecision(12) << x0[0];
	for (int i = 1; i < x0.size(); ++i)
	  out<< setprecision(12) << "\t" << x0[i];
	out << "\n";

	const bool succ = out.good();
	out.close();
	return succ;
  }

  // load the problem from file
  template<typename T>
  inline bool loadQP(Eigen::SparseMatrix<T> &A,Eigen::Matrix<T,-1,1> &B,
					 VVEC4X_T &planes,
					 Eigen::Matrix<T,-1,1> &x0,const string file_name){
	ifstream in;
	in.open(file_name.c_str());
	if (!in.is_open()){
	  ERROR_LOG("failed to open the file: "<<file_name);
	  return false;
	}

	// read dimension
	string temp_str;
	int n, num_planes;
	in >> temp_str >> n >> temp_str >> num_planes;
	assert_ge(n,0);
	assert_ge(num_planes,0);

	A.resize(n,n);
	B.resize(n);
	x0.resize(n);
	planes.resize(num_planes);

	// read A
	int nnz;
	in >> temp_str >> temp_str >> nnz;
	assert_ge(nnz,0);
	A.reserve(nnz);
	std::vector<Eigen::Triplet<T> > tri;
	tri.reserve(nnz);
	for (int i = 0; i < nnz; ++i){
	  int row,col;
	  double value;
	  in >> row >> col >> value;
	  assert_in(row,0,n-1);
	  assert_in(col,0,n-1);
	  tri.push_back(Eigen::Triplet<T>(row,col,value));
	}
	A.setFromTriplets(tri.begin(), tri.end());
	  
	// write B
	in >> temp_str;
	for (int i = 0; i < B.size(); ++i) in >> B[i];

	// write P
	in >> temp_str;
	for (int i = 0; i < planes.size(); ++i){
	  in >> planes[i][0];
	  in >> planes[i][1];
	  in >> planes[i][2];
	  in >> planes[i][3];
	}

	// write x0
	in >> temp_str;
	for (int i = 0; i < x0.size(); ++i) in >> x0[i];

	const bool succ = in.good();
	in.close();
	return succ;
  }
  
}

#endif /* _MPRGPUTILITY_H_ */
