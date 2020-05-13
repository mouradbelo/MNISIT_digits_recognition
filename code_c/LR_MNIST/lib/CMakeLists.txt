#############################
SET ( LIBRARY_NAME   LRLIB )
#############################

#----------------------------------------------------------------------------
# CREATES A USER OPTION IN CMAKE
OPTION ( BUILD_${LIBRARY_NAME}  "Build ${LIBRARY_NAME} library ?" ON)
#----------------------------------------------------------------------------

IF ( BUILD_${LIBRARY_NAME} )
   FILE(GLOB ${LIBRARY_NAME}_HEADERS "*.h" "*.txx")
   FILE(GLOB ${LIBRARY_NAME}_SOURCES *.cxx *.cpp ${${LIBRARY_NAME}_HEADERS})

#   SET ( ${LIBRARY_NAME}_LINK_LIBRARIES
#		${ITK_LIBRARIES}
#		)
 
   ADD_LIBRARY(${LIBRARY_NAME} STATIC  ${${LIBRARY_NAME}_SOURCES})
#   TARGET_LINK_LIBRARIES(${LIBRARY_NAME} ${${LIBRARY_NAME}_LINK_LIBRARIES} )
#   SET_TARGET_PROPERTIES(${LIBRARY_NAME} PROPERTIES LINKER_LANGUAGE C)

   install(TARGETS ${LIBRARY_NAME} DESTINATION lib)
   install(FILES   ${LIBRARY_NAME_HEADERS} DESTINATION include)
 
ENDIF ( BUILD_${LIBRARY_NAME} )
