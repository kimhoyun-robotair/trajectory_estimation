#include <rclcpp/rclcpp.hpp>
#include <nav_msgs/msg/odometry.hpp>
#include <vector>
#include <utility>
#include <cmath>
#include <limits>
#include <string>
#include <algorithm>

// RViz 시각화를 위한 Marker 관련 헤더
#include <visualization_msgs/msg/marker.hpp>
#include <geometry_msgs/msg/point.hpp>

// Kalman 필터 관련 헤더와 Eigen 라이브러리 포함
#include "kalman.hpp"
#include <eigen3/Eigen/Dense>

/**
 * TrajectoryEstimationEgo 노드는 ego odom 데이터를 구독하여
 * odom 데이터를 저장하고 loop closure 이후 칼만 필터로 미래 위치를 예측한 후,
 * RViz2에서 Marker 메시지로 시각화합니다.
 *
 * loop closure 플래그가 true가 된 이후에는
 * 1. 최신 odom 측정값과
 * 2. 저장된 과거 odom 배열에서, 최신 odom 값과 가장 가까운 점의 인덱스 주변 ±5 범위(총 10개)의 평균값
 * 을 결합하여 칼만 필터에 입력으로 사용합니다.
 */
class TrajectoryEstimationEgo : public rclcpp::Node
{
public:
  TrajectoryEstimationEgo() : Node("trajectory_estimation_ego")
  {
    // odom 데이터 관련 변수 초기화
    distance_threshold = 0.2;      // 시작점과의 거리 기준
    min_message_before_check = 10; // loop closure 판정을 위한 최소 메시지 수

    // 칼만 필터 파라미터 초기화 (constant velocity 모델)
    dt_kalman = 1.0 / 30.0;
    A_kalman = Eigen::MatrixXd(4, 4);
    A_kalman << 1, 0, dt_kalman, 0,
                0, 1, 0, dt_kalman,
                0, 0, 1, 0,
                0, 0, 0, 1;
    C_kalman = Eigen::MatrixXd(2, 4);
    C_kalman << 1, 0, 0, 0,
                0, 1, 0, 0;
    Q_kalman = 1e-4 * Eigen::MatrixXd::Identity(4, 4);
    R_kalman = 1e-2 * Eigen::MatrixXd::Identity(2, 2);
    P0_kalman = Eigen::MatrixXd::Identity(4, 4);

    // 칼만 필터 인스턴스 초기화 (ego 전용)
    kalman_filter_ego = KalmanFilter(dt_kalman, A_kalman, C_kalman, Q_kalman, R_kalman, P0_kalman);
    kalman_initialized_ego = false;

    // ego odom 토픽 구독 ("/odom")
    ego_subscription = this->create_subscription<nav_msgs::msg::Odometry>(
      "/odom", 10,
      std::bind(&TrajectoryEstimationEgo::odom_callback, this, std::placeholders::_1));

    // RViz 시각화를 위한 Marker publisher 생성
    marker_pub = this->create_publisher<visualization_msgs::msg::Marker>("visualization_marker", 10);

    // 초기 상태 변수 설정
    ego_starting_initialized = false;
    ego_lap_completed = false;
  }

private:
  // ego odom 콜백 함수: odom 메시지의 frame_id와 stamp를 저장
  void odom_callback(const nav_msgs::msg::Odometry::SharedPtr msg)
  {
    // odom 메시지의 header 정보 저장 (부모 프레임이 "map"이어야 함)
    current_frame_ = msg->header.frame_id;  // 보통 "map"이어야 함.
    current_stamp_ = msg->header.stamp;

    double x = msg->pose.pose.position.x;
    double y = msg->pose.pose.position.y;
    RCLCPP_INFO(this->get_logger(), "Received ego odometry: position=(%.2f, %.2f)", x, y);
    process_odometry(x, y);
  }

  // odom 데이터 처리 함수: loop closure 전에는 데이터를 저장하고, 완료 후 예측 수행
  void process_odometry(double x, double y)
  {
    if (!ego_lap_completed) {
      ego_odom_positions.push_back({x, y});
      if (!ego_starting_initialized) {
        ego_starting_odom = {x, y};
        ego_starting_initialized = true;
        return;
      }
      if (ego_odom_positions.size() >= min_message_before_check) {
        double distanceFromStart = compute_distance({x, y}, ego_starting_odom);
        RCLCPP_INFO(this->get_logger(), "Distance from start (ego): %.2f", distanceFromStart);
        if (distanceFromStart < distance_threshold) {
          ego_lap_completed = true;
          RCLCPP_INFO(this->get_logger(), "Lap completed for ego! Odom data size: %ld", ego_odom_positions.size());
        }
      }
    } else {
      // loop closure 후:
      // 1. 최신 odom 측정값은 (x,y)
      // 2. 저장된 과거 odom 배열에서, 최신 odom과 가장 가까운 점의 인덱스를 구하고, 해당 인덱스 ±5 (총 10개)의 평균을 계산
      std::vector<std::pair<double, double>> predicted = predict_future_positions(x, y, ego_odom_positions, kalman_filter_ego, kalman_initialized_ego);
      predicted_future_ego = predicted;
      publish_predicted_marker(predicted_future_ego);
    }
  }

  // 칼만 필터를 이용한 미래 예측 함수 (최근접 이웃 기준 ±5 범위의 점들 이용하여 두 입력을 결합)
  std::vector<std::pair<double, double>> predict_future_positions(double current_x, double current_y,
                                                                   const std::vector<std::pair<double, double>> &positions,
                                                                   KalmanFilter &kf, bool &kalman_initialized)
  {
    // 현재 최신 odom 값은 (current_x, current_y)

    // 2. 저장된 odom 배열에서 현재 측정값과의 최근접 점의 인덱스를 구합니다.
    auto [nn_index, dummy] = get_nearest_neighbor_index({current_x, current_y}, positions);
    int start_idx = std::max(0, nn_index - 5);
    int end_idx = std::min(static_cast<int>(positions.size()) - 1, nn_index + 5);

    double sum_x = 0.0, sum_y = 0.0;
    int count = 0;
    for (int i = start_idx; i <= end_idx; i++) {
      sum_x += positions[i].first;
      sum_y += positions[i].second;
      count++;
    }
    double avg_window_x = sum_x / count;
    double avg_window_y = sum_y / count;

    // 1번과 2번의 측정값을 결합합니다. (여기서는 단순 평균)
    double combined_x = (current_x + avg_window_x) / 2.0;
    double combined_y = (current_y + avg_window_y) / 2.0;
    RCLCPP_INFO(this->get_logger(), "Combined measurement (ego): x=%.2f, y=%.2f", combined_x, combined_y);

    // 측정값 벡터 구성
    Eigen::VectorXd y_meas(2);
    y_meas << combined_x, combined_y;

    // 칼만 필터 초기화 또는 업데이트
    if (!kalman_initialized) {
      Eigen::VectorXd x0(4);
      x0 << combined_x, combined_y, 0, 0;
      kf.init(0.0, x0);
      kalman_initialized = true;
    } else {
      try {
        kf.update(y_meas);
      } catch (std::runtime_error &e) {
        RCLCPP_ERROR(this->get_logger(), "Kalman filter update error: %s", e.what());
      }
    }

    // 30회의 미래 예측 (dt=1/30초)
    std::vector<std::pair<double, double>> future_predictions;
    Eigen::VectorXd state = kf.getState();
    for (int i = 0; i < 10; i++) {
      state = A_kalman * state;  // 상태 예측: xₖ₊₁ = A * xₖ
      future_predictions.push_back({state(0), state(1)});
    }
    return future_predictions;
  }

  // RViz2에 예측 점들을 Marker 메시지로 발행
  void publish_predicted_marker(const std::vector<std::pair<double, double>> &predicted_points)
  {
    visualization_msgs::msg::Marker marker;
    // odom 토픽의 parent가 "map"이므로, marker의 frame_id도 "map"으로 설정합니다.
    marker.header.frame_id = "map";
    marker.header.stamp = current_stamp_; // 최근 odom 메시지의 stamp 사용
    marker.ns = "predicted_points";
    marker.id = 0;
    marker.type = visualization_msgs::msg::Marker::SPHERE_LIST;
    marker.action = visualization_msgs::msg::Marker::ADD;
    marker.pose.orientation.w = 1.0;

    marker.scale.x = 0.05;
    marker.scale.y = 0.05;
    marker.scale.z = 0.05;

    marker.color.r = 0.0f;
    marker.color.g = 1.0f;
    marker.color.b = 0.0f;
    marker.color.a = 1.0f;

    for (const auto &pt_pair : predicted_points) {
      geometry_msgs::msg::Point pt;
      pt.x = pt_pair.first;
      pt.y = pt_pair.second;
      pt.z = 0.0;
      marker.points.push_back(pt);
    }
    marker_pub->publish(marker);
  }

  // 두 점 사이의 유클리디안 거리 계산
  double compute_distance(const std::pair<double, double>& p1,
                          const std::pair<double, double>& p2)
  {
    double dx = p1.first - p2.first;
    double dy = p1.second - p2.second;
    return std::sqrt(dx * dx + dy * dy);
  }

  // positions 내에서 new_position과의 최단 거리를 가지는 점의 인덱스와 거리를 계산
  std::pair<int, double> get_nearest_neighbor_index(const std::pair<double, double>& new_position,
                                                    const std::vector<std::pair<double, double>> &positions)
  {
    int min_index = -1;
    double min_distance = std::numeric_limits<double>::max();
    for (size_t i = 0; i < positions.size(); i++) {
      double d = compute_distance(new_position, positions[i]);
      if (d < min_distance) {
        min_distance = d;
        min_index = i;
      }
    }
    return {min_index, min_distance};
  }

  // --- 멤버 변수 ---
  // ego odom 관련 구독자 및 publisher
  rclcpp::Subscription<nav_msgs::msg::Odometry>::SharedPtr ego_subscription;
  rclcpp::Publisher<visualization_msgs::msg::Marker>::SharedPtr marker_pub;

  // odom 데이터 저장 및 loop closure 판정을 위한 변수들
  std::vector<std::pair<double, double>> ego_odom_positions;
  std::pair<double, double> ego_starting_odom;
  bool ego_starting_initialized;
  bool ego_lap_completed;
  double distance_threshold;
  size_t min_message_before_check;

  // 칼만 필터 관련 변수 (ego 전용)
  double dt_kalman;
  Eigen::MatrixXd A_kalman;
  Eigen::MatrixXd C_kalman;
  Eigen::MatrixXd Q_kalman;
  Eigen::MatrixXd R_kalman;
  Eigen::MatrixXd P0_kalman;
  KalmanFilter kalman_filter_ego;
  bool kalman_initialized_ego;

  // 미래 예측 결과 저장 (30개씩)
  std::vector<std::pair<double, double>> predicted_future_ego;

  // odom 메시지의 header 정보를 저장 (TF 맞춤용)
  std::string current_frame_;
  rclcpp::Time current_stamp_;
};

int main(int argc, char * argv[])
{
  rclcpp::init(argc, argv);
  auto node = std::make_shared<TrajectoryEstimationEgo>();
  rclcpp::spin(node);
  rclcpp::shutdown();
  return 0;
}
