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
# define debug_fun(event)
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

#include <eigen3/Eigen/Sparse>
#include <eigen3/Eigen/Dense>

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

#define VVEC4X_T std::vector<Eigen::Matrix<T,4,1>, Eigen::aligned_allocator<Eigen::Matrix<T,4,1> > >
#define VVVEC4X_T std::vector<VVEC4X_T >

  template<typename T>
  VVVEC4X_T &convert(const VVEC4X_T &in, VVVEC4X_T &out, const size_t num_nodes){
	  	 
	out.resize(num_nodes);
	for(size_t i = 0; i < num_nodes; i++)
	  out[i] = in;
	return out;
  }

  // convert plane constraints to A and c where: A*x >= c.
  template<typename T, typename VECTOR>
  void convert(const VVVEC4X_T &in, Eigen::SparseMatrix<T> &A, VECTOR &c){

	typedef vector<Eigen::Triplet<T,int> > TRIPS;
	TRIPS trips;
	std::vector<T> rhs;
	trips.reserve(in.size()*3);
	rhs.reserve(in.size());

	for (size_t vert_id = 0; vert_id < in.size(); ++vert_id){
	  for (size_t plane_id = 0; plane_id < in[vert_id].size(); ++plane_id){
		const T p = in[vert_id][plane_id][3];
		const Eigen::Matrix<T,3,1> n = in[vert_id][plane_id].head(3);
		const int row = rhs.size();
		const int col0 = vert_id*3;
		trips.push_back( Eigen::Triplet<T,int>(row, col0+0, n[0]) );
		trips.push_back( Eigen::Triplet<T,int>(row, col0+1, n[1]) );
		trips.push_back( Eigen::Triplet<T,int>(row, col0+2, n[2]) );
		rhs.push_back(-p);
	  }
	}
	
	const int num_var = in.size()*3;
	A.setZero();
	A.resize(rhs.size(), num_var);
	A.reserve(trips.size());
	A.setFromTriplets( trips.begin(), trips.end() );
	A.makeCompressed();
  
	c.resize(rhs.size());
	for (int i = 0; i < c.size(); ++i){
	  c[i] = rhs[i];
	}
  }

  // write and load sparse matrix as binay file
  template<typename T>
  inline bool writeSparseMatrix(ofstream &out, const Eigen::SparseMatrix<T> &A){

	const size_t A_rows = A.rows();
	const size_t A_cols = A.cols();
	const size_t A_nz = A.nonZeros();
	out.write((char*)&(A_rows),sizeof(A_rows));
	out.write((char*)&(A_cols),sizeof(A_cols));
	out.write((char*)&(A_nz),sizeof(A_nz));

	std::vector<Eigen::Triplet<T> > A_data;
	A_data.reserve(A.nonZeros());
	for ( int k = 0; k < A.outerSize(); ++k ){
	  for ( typename Eigen::SparseMatrix<T>::InnerIterator it(A,k); it; ++it )
		A_data.push_back(Eigen::Triplet<T>(it.row(), it.col(), it.value()));
	}
	if (A_data.size() > 0)
	  out.write((char*)&A_data[0],sizeof(Eigen::Triplet<T>)*A_nz);
	return out.good();
  }

  template<typename T>
  inline bool loadSparseMatrix(ifstream &in, Eigen::SparseMatrix<T> &A){

	size_t rows = 0;
	size_t cols = 0;
	size_t nnz = 0;
	in.read((char*)&(rows),sizeof(rows));
	in.read((char*)&(cols),sizeof(cols));
	in.read((char*)&(nnz),sizeof(nnz));

	DEBUG_LOG("rows = " << rows);
	DEBUG_LOG("cols = " << cols);
	DEBUG_LOG("nnz = " << nnz);

	A.setZero();
	A.resize(rows, cols);
	if(nnz > 0){
	  A.reserve(nnz);
	  std::vector<Eigen::Triplet<T> > tri(nnz);
	  in.read((char*)&tri[0], sizeof(Eigen::Triplet<T>)*tri.size());
	  A.setFromTriplets(tri.begin(), tri.end());
	}else{
	  A.setZero();
	}
	return in.good();
  }

  // write and load the problem: A, B, x0 and the constraints, i.e planes.
  // the problem is: 
  // min_{x} 1/2*x^t*A*x-x^t*B s.t. n_i*x_j+p_i>= 0
  template<typename T>
  inline bool writeQP(const Eigen::SparseMatrix<T> &A,const Eigen::Matrix<T,-1,1> &B,
					  const VVEC4X_T &planes,
					  const Eigen::Matrix<T,-1,1> &x0,const string file_name){

	// open file
	ofstream out;
	out.open(file_name.c_str());
	if (!out.is_open()){
	  ERROR_LOG("failed to open the file: "<<file_name);
	  return false;
	}

	// write dimensions
	out << "dimension " << A.rows() << "\n";
	out << "planes "<< planes.size() << "\n";
	out << "A\n";
	out << "non_zeros "<< A.nonZeros() << "\n";

	// write A
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

  template<typename T>
  inline bool writeQP(const Eigen::SparseMatrix<T> &A,const Eigen::Matrix<T,-1,1> &B,
					  const VVVEC4X_T &planes_for_each_node,
					  const Eigen::Matrix<T,-1,1> &x0,const string file_name){

	// open file
	ofstream out(file_name.c_str(), ios::out|ios::binary);
	if (!out.is_open()){
	  ERROR_LOG("failed to open the file: "<<file_name);
	  return false;
	}
	
	// write A
	writeSparseMatrix(out, A);
	  
	// write B
	assert_eq(B.size(), A.rows());
	out.write((char*)&B[0],sizeof(T)*B.size());

	// write P
	assert_eq(planes_for_each_node.size()*3,A.rows());
	for (size_t i = 0; i < planes_for_each_node.size(); ++i){
	  const size_t p = planes_for_each_node[i].size();
	  out.write( (char*)&p, sizeof(p) );
	  for (size_t j = 0; j < p; ++j)
		out.write((char*)&(planes_for_each_node[i][j][0]), sizeof(T)*4);
	}

	// write x0
	assert_eq(x0.size(), A.rows());
	out.write((char*)&x0[0],sizeof(T)*x0.size());

	const bool succ = out.good();
	out.close();
	return succ;
  }

  template<typename T>
  inline bool loadQP(Eigen::SparseMatrix<T> &A,Eigen::Matrix<T,-1,1> &B,
					 VVEC4X_T &planes,
					 Eigen::Matrix<T,-1,1> &x0,const string file_name){

	// open file
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
	  T value;
	  in >> row >> col >> value;
	  assert_in(row,0,n-1);
	  assert_in(col,0,n-1);
	  tri.push_back(Eigen::Triplet<T>(row,col,value));
	}
	A.setFromTriplets(tri.begin(), tri.end());
	  
	// read B
	in >> temp_str;
	for (int i = 0; i < B.size(); ++i) in >> B[i];

	// read P
	in >> temp_str;
	for (int i = 0; i < planes.size(); ++i){
	  in >> planes[i][0];
	  in >> planes[i][1];
	  in >> planes[i][2];
	  in >> planes[i][3];
	}

	// read x0
	in >> temp_str;
	for (int i = 0; i < x0.size(); ++i) in >> x0[i];

	const bool succ = in.good();
	in.close();
	return succ;
  }

  template<typename T>
  inline bool loadQP(Eigen::SparseMatrix<T> &A,Eigen::Matrix<T,-1,1> &B,
					 VVVEC4X_T &planes_for_each_node,
					 Eigen::Matrix<T,-1,1> &x0,const string file_name){

	// open file
	ifstream in(file_name.c_str(), ios::in|ios::binary);
	if (!in.is_open()){
	  ERROR_LOG("failed to open the file: "<<file_name);
	  return false;
	}

	// read A
	loadSparseMatrix(in, A);
	const int rows = A.rows();
	  
	// read B
	B.resize(A.rows());
	if(B.size() > 0){
	  in.read((char*)&B[0],sizeof(T)*B.size());
	}

	// read P
	assert_eq(rows%3,0);
	planes_for_each_node.resize(rows/3);
	for (size_t i = 0; i < planes_for_each_node.size(); ++i){
	  size_t p = 0;
	  in.read((char*)&p, sizeof(p));
	  planes_for_each_node[i].resize(p);
	  for (size_t j = 0; j < p; ++j)
		in.read((char*)&(planes_for_each_node[i][j][0]), sizeof(T)*4);
	}

	// read x0
	x0.resize(A.rows());
	if(x0.size() > 0){
	  in.read((char*)&x0[0],sizeof(T)*x0.size());
	}

	const bool succ = in.good();
	in.close();
	return succ;

  }

  // write and load QP problem:
  // A*x = B   s.t.  J*x >= c, where A and J are sparse matrices, 
  // and x0 is the inital value.
  template<typename T>
  inline bool writeQP(const Eigen::SparseMatrix<T> &A,const Eigen::Matrix<T,-1,1> &B,
					  const Eigen::SparseMatrix<T> &J,const Eigen::Matrix<T,-1,1> &c,
					  const Eigen::Matrix<T,-1,1> &x0,const string file_name){

	// open file
	ofstream out(file_name.c_str(), ios::out|ios::binary);
	if (!out.is_open()){
	  ERROR_LOG("failed to open the file: "<<file_name);
	  return false;
	}
	
	// write A
	writeSparseMatrix(out, A);
	  
	// write B
	assert_eq(B.size(), A.rows());
	if(B.size() > 0)
	  out.write((char*)&B[0],sizeof(T)*B.size());

	// write J
	writeSparseMatrix(out, J);

	// write c
	assert_eq(c.size(), J.rows());
	if (c.size() > 0)
	  out.write((char*)&c[0],sizeof(T)*c.size());

	// write x0
	assert_eq(x0.size(), A.rows());
	if (x0.size() > 0)
	  out.write((char*)&x0[0],sizeof(T)*x0.size());

	const bool succ = out.good();
	out.close();
	return succ;
  }

  template<typename T>
  inline bool loadQP(Eigen::SparseMatrix<T> &A,Eigen::Matrix<T,-1,1> &B,
					 Eigen::SparseMatrix<T> &J,Eigen::Matrix<T,-1,1> &c,
					 Eigen::Matrix<T,-1,1> &x0,const string file_name){

	// open file
	ifstream in(file_name.c_str(), ios::in|ios::binary);
	if (!in.is_open()){
	  ERROR_LOG("failed to open the file: "<<file_name);
	  return false;
	}

	// read A
	loadSparseMatrix(in, A);
	  
	// read B
	B.resize(A.rows());
	if(B.size() > 0){
	  in.read((char*)&B[0],sizeof(T)*B.size());
	}

	// read J
	loadSparseMatrix(in, J);

	// read c
	c.resize(J.rows());
	if(c.size() > 0){
	  in.read((char*)&c[0],sizeof(T)*c.size());
	}

	// read x0
	x0.resize(A.rows());
	if(x0.size() > 0){
	  in.read((char*)&x0[0],sizeof(T)*x0.size());
	}

	const bool succ = in.good();
	in.close();
	return succ;

  }

  // check the validation of the lagragian multipliers
  template<typename T>
  inline bool greaterThan(const vector<vector<T> > &all_lambdas, const T tol=0.0f){
	  
	bool valid = true;
	for (size_t i = 0; i < all_lambdas.size() && valid; ++i){
	  const vector<T> &lambdas = all_lambdas[i];
	  for (size_t p = 0; p < lambdas.size() && valid; ++p){
		ERROR_LOG_COND("i = "<<i<<", p = "<<p <<", tol: "<<tol <<"\nlambda = "<<lambdas[p]<<"\n",(lambdas[p]>=tol));
		valid = (lambdas[p] >= tol);
	  }
	}
	return valid;
  }

  
  template<typename T, typename Vec>
  inline void getDiagonal(const Eigen::SparseMatrix<T> &A, Vec &diag_A){
	
	diag_A.resize( min( A.rows(), A.cols() ) );
	for(int k = 0; k < A.outerSize(); ++k)
	  for(typename Eigen::SparseMatrix<T>::InnerIterator it(A,k);it;++it){
		if (it.col() == it.row()){
		  assert_eq(it.row(),k);
		  diag_A[k] = it.value();
		  break;
		}
	  }
  }

}

#endif /* _MPRGPUTILITY_H_ */
