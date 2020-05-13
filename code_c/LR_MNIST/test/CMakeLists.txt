

#----------------------------------------------------------------------------
# CREATES A USER OPTION IN CMAKE
OPTION ( BUILD_TESTS "Build Tests ?" ON)
#----------------------------------------------------------------------------

IF ( BUILD_TESTS )
	add_executable(test_common_functions test_common_functions.cpp)
	target_link_libraries(test_common_functions LRLIB )
	install(TARGETS test_common_functions DESTINATION bin)
	
       # add_executable(test_common_functions_time test_common_functions_time.cpp)
       # target_link_libraries(test_common_functions_time LRLIB )
       # install(TARGETS test_common_functions_time DESTINATION bin)


ENDIF ( BUILD_TESTS )
