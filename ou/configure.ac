#                                               -*- Autoconf -*-
# Process this file with autoconf to produce a configure script.

AC_PREREQ(2.61)
AC_INIT(ou, 0, oleh_derevenko@users.sourceforge.net)
AC_CONFIG_SRCDIR([src/ou/atomic.cpp])
# AC_CONFIG_HEADER([config.h])
AC_CANONICAL_HOST
AC_USE_SYSTEM_EXTENSIONS
AM_INIT_AUTOMAKE(foreign)

AC_MSG_CHECKING([target OS])
case "$host_os" in
  cygwin* | mingw*)
    targetos=_OU_TARGET_OS_WINDOWS
    CXXFLAGS+="-mthreads $CXXFLAGS"
    AC_MSG_RESULT([win32])
    ;;
  *qnx*)
    targetos=_OU_TARGET_OS_QNX
    AC_MSG_RESULT([qnx])
    ;;
  *apple* | *darwin*)
    targetos=_OU_TARGET_OS_MAC
    AC_MSG_RESULT([darwin])
    ;;
  *sunos*)
    targetos=_OU_TARGET_OS_SUNOS
    AC_MSG_RESULT([sunos])
    ;;
  *aix*)
    targetos=_OU_TARGET_OS_AIX
    AC_MSG_RESULT([aix])
    ;;
  *)
    targetos=_OU_TARGET_OS_GENUNIX
    AC_MSG_RESULT([unix])
    ;;
esac


#echo "host OS name: $host_os"
#TODO: _OU_TARGET_BITS ?



# Checks for programs.
AC_PROG_CXX
AC_PROG_CC
AC_PROG_AWK
AC_PROG_INSTALL
AC_PROG_RANLIB
AC_PROG_CPP
AC_PROG_MKDIR_P
AC_LIBTOOL_WIN32_DLL
AC_PROG_LIBTOOL


# Checks for libraries.
AC_CHECK_LIB([c], main) # needed for sunos?
AC_CHECK_LIB([pthread], [main])
AC_DEFINE(_REENTRANT,1,[enable thread-safe functions])



# Checks for header files.
AC_HEADER_STDC
AC_CHECK_HEADERS([inttypes.h malloc.h stddef.h stdlib.h string.h])

# Checks for typedefs, structures, and compiler characteristics.
AC_HEADER_STDBOOL
AC_C_CONST
AC_C_INLINE
AC_TYPE_INT16_T
AC_TYPE_INT32_T
AC_TYPE_INT64_T
AC_TYPE_INT8_T
AC_TYPE_SIZE_T
AC_TYPE_UINT16_T
AC_TYPE_UINT32_T
AC_TYPE_UINT64_T
AC_TYPE_UINT8_T
AC_C_VOLATILE
AC_CHECK_TYPES([ptrdiff_t])

# Checks for library functions.
AC_FUNC_MALLOC
AC_FUNC_REALLOC
AC_CHECK_FUNCS([memset])

if test $targetos = _OU_TARGET_OS_MAC
then
    MAC_OS_X_VERSION=1000
    AC_CHECK_FUNC([OSAtomicAdd32Barrier], [MAC_OS_X_VERSION=1040])
    AC_CHECK_FUNC([OSAtomicAnd32OrigBarrier], [MAC_OS_X_VERSION=1050])
    AC_DEFINE_UNQUOTED(MAC_OS_X_VERSION, $MAC_OS_X_VERSION, [Mac OS X version])
fi

if test $targetos = _OU_TARGET_OS_SUNOS
then
    AC_CHECK_FUNC(atomic_inc_32_nv, [],
        [targetos=_OU_TARGET_OS_GENUNIX])
fi

AC_DEFINE_UNQUOTED(_OU_TARGET_OS, $targetos)


############################################
# NVIDIA Cuda Compiler detection and setup #
############################################

# If cuda is requested to be enabled
AC_ARG_ENABLE(cuda, 
        AS_HELP_STRING([--enable-cuda=ARCH], [Enable cuda based modules for architecture ARCH (see nvcc option -arch).]),[

        # Search nvcc compiler
        AC_PATH_PROG(NVCC, nvcc, "no")
        AS_IF([test "x$NVCC" = "xno"],[
                        AC_MSG_ERROR([NVCC compiler not found!])
        ])

        # Check nvcc version, should be 3.0
        AS_IF([nvcc --version | grep -q "release 3.0"],
                        [],
                        [AC_MSG_WARN([NVCC compiler version is NOT 3.0!])
        ])

        # If $with_cuda is not empty, set to CUDA_ARCH to
        # supplied value, else set to value sm_11
        AS_IF([test "x$enableval" = "xyes"],[
                        CUDA_ARCH=" -arch=sm_11"
        ],[
                CUDA_ARCH=" -arch=$enableval"
        ])

                # Set CUDA_CFLAGS to $NVCC, where substring "bin/nvcc"
                # is substituted by "include".
                CUDA_CFLAGS=" -I${NVCC/'bin/nvcc'/include}"

                #Set CUDA_CFLAGS to $NVCC, where substring "bin/nvcc"
                #is substituted by "lib".
                CUDA_LIBS=" -L${NVCC/'bin/nvcc'/lib}"

                # If $build_cpu contains "_64", append "64" to CUDA_LIBS
        AS_IF([echo $build_cpu | grep -q "_64"],
                [CUDA_LIBS+="64"])

                # Append " -lcudart" to CUDA_LIBS
                CUDA_LIBS+=" -lcudart"

        # Symbolize that cuda is wanted
        with_cuda=$enableval

        # Make variables available in Makefile.am
        AC_SUBST(CUDA_CFLAGS)
        AC_SUBST(CUDA_LIBS)
        AC_SUBST(NVCC)
])
# Set this conditional if cuda is wanted
AM_CONDITIONAL([WANT_CUDA], [test -n "$with_cuda"])

# Check whether to use device emulation mode for cuda (if no cuda capable gpu is available)
AC_ARG_ENABLE([emu],
        AS_HELP_STRING([--enable-emu], [Enable device emulation for cuda modules (nvcc version <= 3.0 only).]),
        [EMULATION=true],
        [EMULATION=false])

# Set up compilation flags for cuda compiler nvcc, if with_cuda is set
AS_IF([test -n "$with_cuda"],[

        # If debug flag is set apply debugging compilation flags, otherwise build compilation flags
        AS_IF([test "x$DEBUG" = "xtrue"],
                [NVCCFLAGS="-g --compiler-options -fno-strict-aliasing --compiler-options -fno-inline"],
                [NVCCFLAGS="-O3 -use_fast_math --compiler-options -fno-strict-aliasing --compiler-options -fno-inline"])

        # Add architecture to flags
        NVCCFLAGS+=" $CUDA_ARCH"

        # If device emulation was set, add deviceemu flag
        AS_IF([test "x$EMULATION" = "xtrue"],
                [NVCCFLAGS+=" -deviceemu"])

])
# Make NVCCFLAGS available in Makefile.am
AC_SUBST(NVCCFLAGS)

##################################################################################################

AC_ARG_VAR([OU_NAMESPACE], [which namespace OU will be compiled in])
AC_ARG_WITH([namespace],
    AC_HELP_STRING([--with-namespace=name],[sets the namespace for compiled code]),
    [OU_NAMESPACE=$withval])
if test x$OU_NAMESPACE = xno -o x$OU_NAMESPACE = x
then
    OU_NAMESPACE="ou"
fi
CPPFLAGS="$CPPFLAGS -D_OU_NAMESPACE=$OU_NAMESPACE"

AC_ARG_ENABLE([asserts],
        AS_HELP_STRING([--disable-asserts],
            [disables debug error checking]),
        asserts=$enableval,asserts=yes)
if test x$asserts = xno
then
    CPPFLAGS="$CPPFLAGS -DNDEBUG"
fi

AC_CONFIG_FILES([Makefile
                 include/ou/Makefile
                 src/ou/Makefile
                 test/Makefile])
AC_OUTPUT

echo "OU namespace: $OU_NAMESPACE"

