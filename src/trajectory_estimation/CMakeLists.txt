cmake_minimum_required(VERSION 3.8)
project(trajectory_estimation)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# find dependencies
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(nav_msgs REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(visualization_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)

add_executable(trajectory_estimation src/trajectory_estimation.cpp src/kalman-cpp/kalman.cpp)
ament_target_dependencies(trajectory_estimation rclcpp nav_msgs Eigen3 visualization_msgs geometry_msgs)

add_executable(code_test src/test.cpp src/kalman-cpp/kalman.cpp)
ament_target_dependencies(code_test rclcpp nav_msgs Eigen3 visualization_msgs geometry_msgs)

install(TARGETS trajectory_estimation code_test
  DESTINATION lib/${PROJECT_NAME}
)

install(
  DIRECTORY launch config
  DESTINATION share/${PROJECT_NAME}
)

target_include_directories(trajectory_estimation PUBLIC
  ${CMAKE_CURRENT_SOURCE_DIR}/src/kalman-cpp
)

target_include_directories(code_test PUBLIC
  ${CMAKE_CURRENT_SOURCE_DIR}/src/kalman-cpp
)

if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  # the following line skips the linter which checks for copyrights
  # comment the line when a copyright and license is added to all source files
  set(ament_cmake_copyright_FOUND TRUE)
  # the following line skips cpplint (only works in a git repo)
  # comment the line when this package is in a git repo and when
  # a copyright and license is added to all source files
  set(ament_cmake_cpplint_FOUND TRUE)
  ament_lint_auto_find_test_dependencies()
endif()

ament_package()
