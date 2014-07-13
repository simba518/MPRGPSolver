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

#include <omp.h>

#ifdef _MSC_VER
#define STRINGIFY(X) X
#define PRAGMA __pragma
#else
#define STRINGIFY(X) #X
#define PRAGMA _Pragma
#endif

#define OMP_PARALLEL_FOR_ PRAGMA(STRINGIFY(omp parallel for num_threads(OmpSettings::getOmpSettings().nrThreads()) schedule(dynamic,OmpSettings::getOmpSettings().szChunk())))
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
}

#endif /* _MPRGPUTILITY_H_ */
