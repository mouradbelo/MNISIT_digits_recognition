

#----------------------------------------------------------------------------
# CREATES A USER OPTION IN CMAKE
OPTION ( BUILD_SRC  "Build applications in /src ?" ON)
#----------------------------------------------------------------------------

IF ( BUILD_SRC )
	add_executable(mnist_train_lrgd mnist_train_lrgd.cpp)
	target_link_libraries(mnist_train_lrgd LRLIB )
	install(TARGETS mnist_train_lrgd DESTINATION bin)

    add_executable(mnist_test mnist_test.cpp)
    target_link_libraries(mnist_test LRLIB )
    install(TARGETS mnist_test DESTINATION bin)

	add_executable(mnist_train_lrCgd mnist_train_lrCgd.cpp)
	target_link_libraries(mnist_train_lrCgd LRLIB )
	install(TARGETS mnist_train_lrCgd DESTINATION bin)
	
ENDIF ( BUILD_SRC )
