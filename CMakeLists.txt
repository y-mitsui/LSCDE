cmake_minimum_required (VERSION 2.6)

project (LSCDE)

SET(CMAKE_C_FLAGS "-Wall -g")

ADD_LIBRARY(lscde STATIC LSCDE.c)

add_executable (conditionalProbability LSCDE.c)
target_link_libraries(conditionalProbability m meschach)

# 共有ライブラリのインストール指定
INSTALL(TARGETS lscde ARCHIVE DESTINATION lib)
# 共有ライブラリ用ヘッダーファイルのインストール指定
INSTALL(FILES lscde.h DESTINATION include)
