set(main_IMPLEMENTATIONS
  argos_mng.cpp
  )

add_executable(argos_mng ${main_IMPLEMENTATIONS})
target_link_libraries(argos_mng
  kilobot_rw
  argos3core_simulator
  argos3plugin_simulator_dynamics2d
  argos3plugin_simulator_entities
  argos3plugin_simulator_kilobot
  argos3plugin_simulator_genericrobot
  argos3plugin_simulator_media)
