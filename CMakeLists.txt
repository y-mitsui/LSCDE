cmake_minimum_required (VERSION 2.6)

project (LSCDE)

SET(CMAKE_C_FLAGS "-Wall -g -O0")

ADD_LIBRARY(lscde STATIC lscde.c)
target_link_libraries(lscde m meschach)

add_executable (conditionalProbability conditionalProbability.c)
target_link_libraries(conditionalProbability m lscde)

# 共有ライブラリのインストール指定
INSTALL(TARGETS lscde ARCHIVE DESTINATION lib)
# 共有ライブラリ用ヘッダーファイルのインストール指定
INSTALL(FILES lscde.h DESTINATION include)
