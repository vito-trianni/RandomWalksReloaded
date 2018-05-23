cmake_minimum_required(VERSION 2.8.12)

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${CMAKE_SOURCE_DIR}/cmake)
set(kilobot_rw_SOURCES 
  kilobot_rw_ctrl.cpp
  kilobot_rw_lf.cpp
  )
find_package(Lua52 REQUIRED)

add_library(kilobot_rw SHARED ${kilobot_rw_SOURCES})
target_link_libraries(kilobot_rw
  argos3core_simulator
  argos3plugin_simulator_dynamics2d
  argos3plugin_simulator_entities
  argos3plugin_simulator_kilobot
  argos3plugin_simulator_genericrobot
  argos3plugin_simulator_media)

if(ARGOS_COMPILE_QTOPENGL)
  target_link_libraries(kilobot_rw argos3plugin_simulator_qtopengl)
endif(ARGOS_COMPILE_QTOPENGL)
