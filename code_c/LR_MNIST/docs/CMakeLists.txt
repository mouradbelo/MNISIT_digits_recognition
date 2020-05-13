#----------------------------------------------------------------------------
# CREATES A USER OPTION IN CMAKE
OPTION ( BUILD_DOC "Build documentation ?" OFF )
#----------------------------------------------------------------------------

IF ( BUILD_DOC )
# check if Doxygen is installed
	find_package(Doxygen)
	IF (DOXYGEN_FOUND)
		# set input and output files
		set(DOXYGEN_IN ${CMAKE_CURRENT_SOURCE_DIR}/Doxyfile.in)
		set(DOXYGEN_OUT ${CMAKE_CURRENT_BINARY_DIR}/Doxyfile)

		# request to configure the file
		configure_file(${DOXYGEN_IN} ${DOXYGEN_OUT} @ONLY)
		message("Doxygen build started")

		# note the option ALL which allows to build the docs together with the application
		add_custom_target( doc_doxygen ALL
			COMMAND ${DOXYGEN_EXECUTABLE} ${DOXYGEN_OUT}
			WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
			COMMENT "Generating API documentation with Doxygen"
			VERBATIM )
	ELSE (DOXYGEN_FOUND)
		message("Doxygen need to be installed to generate the doxygen documentation")
	ENDIF (DOXYGEN_FOUND)

	install(DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/html DESTINATION doc)

ENDIF ( BUILD_DOC )