# install 
SET(HEADFILE_DEST /usr/local/include/MPRGP)
FILE(GLOB header ${header} *.h)
INSTALL(FILES ${header} DESTINATION ${HEADFILE_DEST})