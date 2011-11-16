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

# -------------------------------------------------------------
# Setup CUDA paths
# -------------------------------------------------------------
AC_ARG_WITH([cuda],
    [  --with-cuda=PATH			prefix where cuda is installed [default=auto]])
if test -n "$with_cuda"
then
    CUDA_CFLAGS="-I$with_cuda/include"
    CUDA_LIBS="-L$with_cuda/lib -lcuda -lcudart"
    NVCC="$with_cuda/bin/nvcc"
else
    CUDA_CFLAGS="-I/usr/local/cuda/include"
    CUDA_LIBS="-L/usr/local/cuda/lib -lcuda -lcudart"
    NVCC="nvcc"
fi
AC_SUBST(CUDA_CFLAGS)
AC_SUBST(CUDA_LIBS)
AC_SUBST(NVCC)

AC_ARG_ENABLE([emu],
    [  --enable-emu    Turn on device emulation for CUDA],
    [case "${enableval}" in
		yes) EMULATION=true;;
		no)  EMULATION=false;;
		*) AC_MSG_ERROR([bad value ${enableval} for --enable-emu]);;
    esac],
    [EMULATION=false]
)

# -------------------------------------------------------------
# Setup nvcc flags
# -------------------------------------------------------------
if test x$DEBUG = xtrue
then
    NVCCFLAGS="-g"
else
    NVCCFLAGS="-O3 -use_fast_math"
fi
if test x$EMULATION = xtrue
then
    NVCCFLAGS+=" -deviceemu"
fi
AC_SUBST(NVCCFLAGS)

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

