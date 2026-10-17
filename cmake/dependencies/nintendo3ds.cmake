#=================== ImGui ===================
# No OpenGL on the 3DS; the GUI runs headless (see LUS_HEADLESS_GUI), but Fast3dGui still
# references the SDL2 platform backend, so ImGui links SDL2.
find_package(SDL2 REQUIRED)
target_link_libraries(ImGui PUBLIC SDL2::SDL2-static)
