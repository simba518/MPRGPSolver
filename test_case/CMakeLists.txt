SET(APP_NAME test_mprgp)

AUX_SOURCE_DIRECTORY(. test_mprgp_src)

ADD_EXECUTABLE(${APP_NAME} 
  ${test_mprgp_src}
  )

TARGET_LINK_LIBRARIES(${APP_NAME}
  
  ${Boost_LIBRARIES}
  )