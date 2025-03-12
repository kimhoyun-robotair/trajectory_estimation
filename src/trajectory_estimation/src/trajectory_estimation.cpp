#include <rclcpp/rclcpp.hpp>
#include <nav_msgs/msg/odometry.hpp>
#include <vector>
#include <utility>
#include <cmath>
#include <limits>
#include <string>
#include <algorithm> // for std::max, std::min

// RViz 시각화를 위한 Marker 관련 헤더
#include <visualization_msgs/msg/marker.hpp>
#include <geometry_msgs/msg/point.hpp>

// Kalman 필터 관련 헤더와 Eigen 라이브러리 포함
#include "kalman.hpp"
#include <eigen3/Eigen/Dense>

/**
 * TrajectoryEstimation 노드는 ego odom을 구독하여 저장하고,
 * loop closure 이후 칼만 필터를 이용해 미래 odom 값을 예측하며,
 * ego odom에 대해 예측한 미래 위치들을 RViz2에서 Marker 메시지로 시각화합니다.
 */
class TrajectoryEstimation : public rclcpp::Node
{
public:
  TrajectoryEstimation() : Node("trajectory_estimation")
  {
    // 1. 새로운 threshold 멤버 변수 collision_threshold_dist를 0.02로 초기화
    collision_threshold_dist = 0.02;

    // ego odom 토픽 구독
    ego_subscription = this->create_subscription<nav_msgs::msg::Odometry>(
      "/odom", 10,
      std::bind(&TrajectoryEstimation::odom_callback, this, std::placeholders::_1));

    // opponent odom 토픽 구독 (여기서는 신경쓰지 않음)
    opponent_subscription = this->create_subscription<nav_msgs::msg::Odometry>(
      "/opponent_odom", 10,
      std::bind(&TrajectoryEstimation::opponent_odom_callback, this, std::placeholders::_1));

    // Marker publisher 생성 (RViz2에서 예측 점들을 시각화)
    marker_pub = this->create_publisher<visualization_msgs::msg::Marker>("visualization_marker", 10);

    // 공통 변수 초기화
    distance_threshold = 0.2;      // 저장된 odom의 첫 원소와의 거리가 0.2m 이하이면 loop closure로 판단
    min_message_before_check = 10; // loop closure 판단을 위한 최소 메시지 수

    // ego, opponent 별 상태 변수 초기화
    ego_starting_initialized = false;
    ego_lap_completed = false;
    opponent_starting_initialized = false;
    opponent_lap_completed = false;

    // 칼만 필터를 위한 constant velocity 모델 매개변수 설정
    dt_kalman = 1.0 / 30.0;  // 시간 간격: 1/30초

    // 상태 전이 행렬 A (4x4): [ x, y, vx, vy ] 모델 (constant velocity)
    A_kalman = Eigen::MatrixXd(4, 4);
    A_kalman << 1, 0, dt_kalman, 0,
                0, 1, 0, dt_kalman,
                0, 0, 1, 0,
                0, 0, 0, 1;

    // 측정 행렬 C (2x4): x, y 측정만 사용
    C_kalman = Eigen::MatrixXd(2, 4);
    C_kalman << 1, 0, 0, 0,
                0, 1, 0, 0;

    // 프로세스 노이즈 행렬 Q
    Q_kalman = 1e-4 * Eigen::MatrixXd::Identity(4, 4);
    // 측정 노이즈 행렬 R
    R_kalman = 1e-2 * Eigen::MatrixXd::Identity(2, 2);
    // 초기 오차 공분산 행렬 P0
    P0_kalman = Eigen::MatrixXd::Identity(4, 4);

    // 칼만 필터 인스턴스 초기화 (상태는 나중에 초기 측정값으로 설정)
    kalman_filter_ego = KalmanFilter(dt_kalman, A_kalman, C_kalman, Q_kalman, R_kalman, P0_kalman);
    kalman_filter_opp = KalmanFilter(dt_kalman, A_kalman, C_kalman, Q_kalman, R_kalman, P0_kalman);
    kalman_initialized_ego = false;
    kalman_initialized_opp = false;
  }

private:
  // --- Callback 함수들 ---
  // ego odom 콜백 함수
  void odom_callback(const nav_msgs::msg::Odometry::SharedPtr msg)
  {
    double x = msg->pose.pose.position.x;
    double y = msg->pose.pose.position.y;
    RCLCPP_INFO(this->get_logger(), "Received ego odometry: position=(%.2f, %.2f)", x, y);

    process_odometry(x, y,
                     ego_odom_positions,
                     ego_starting_odom,
                     ego_starting_initialized,
                     ego_lap_completed,
                     "ego");
  }

  // opponent odom 콜백 함수 (여기서는 기존 코드 그대로 두지만, 시각화는 ego만 사용)
  void opponent_odom_callback(const nav_msgs::msg::Odometry::SharedPtr msg)
  {
    double x = msg->pose.pose.position.x;
    double y = msg->pose.pose.position.y;
    RCLCPP_INFO(this->get_logger(), "Received opponent odometry: position=(%.2f, %.2f)", x, y);

    process_odometry(x, y,
                     opponent_odom_positions,
                     opponent_starting_odom,
                     opponent_starting_initialized,
                     opponent_lap_completed,
                     "opponent");
  }

  // --- 공통 odometry 처리 함수 ---
  // loop closure 전에는 odom 데이터를 저장하고, loop closure 후에는
  // 가장 가까운 점 계산 및 미래 예측(칼만 필터 업데이트 및 예측) 수행
  void process_odometry(double x, double y,
                        std::vector<std::pair<double, double>> &positions,
                        std::pair<double, double> &starting_odom,
                        bool &starting_initialized,
                        bool &lap_completed,
                        const std::string &odom_type)
  {
    if (!lap_completed) {
      positions.push_back({x, y});
      if (!starting_initialized) {
        starting_odom = {x, y};
        starting_initialized = true;
        return;
      }
      if (positions.size() >= min_message_before_check) {
        double distanceFromStart = compute_distance({x, y}, starting_odom);
        RCLCPP_INFO(this->get_logger(), "Distance from start (%s): %.2f", odom_type.c_str(), distanceFromStart);
        if (distanceFromStart < distance_threshold) {
          lap_completed = true;
          RCLCPP_INFO(this->get_logger(),
                      "Lap completed for %s! Stopping odom collection. Odom data size: %ld",
                      odom_type.c_str(), positions.size());
        }
      }
    } else {
      // loop closure 완료 후: 가장 가까운 이웃 인덱스 계산
      auto [index, nn_distance] = get_nearest_neighbor_index({x, y}, positions);
      RCLCPP_INFO(this->get_logger(), "Nearest neighbor index for %s: %d, distance: %.2f",
                  odom_type.c_str(), index, nn_distance);

      // 미래 예측 수행 (칼만 필터 업데이트 및 10회 예측)
      if (odom_type == "ego") {
        std::vector<std::pair<double, double>> predicted = predict_future_positions(x, y, positions, kalman_filter_ego, kalman_initialized_ego);
        predicted_future_ego = predicted;
        // ego 예측 결과를 RViz2로 시각화 (Marker publish)
        publish_predicted_marker(predicted_future_ego);
        // opponent와의 충돌 판정은 기존 코드에서 수행 (생략 가능)
        if (!predicted_future_opponent.empty()) {
          check_collision(predicted, predicted_future_opponent);
        }
      } else if (odom_type == "opponent") {
        std::vector<std::pair<double, double>> predicted = predict_future_positions(x, y, positions, kalman_filter_opp, kalman_initialized_opp);
        predicted_future_opponent = predicted;
        if (!predicted_future_ego.empty()) {
          check_collision(predicted_future_ego, predicted);
        }
      }
    }
  }

  // --- 칼만 필터를 이용한 미래 예측 함수 ---
  // 최근접 이웃 인덱스 기준 ±5 범위의 점들을 추출한 후,
  // 현재 측정값과 함께 평균을 계산하여 칼만 필터의 입력 측정값으로 사용하고,
  // dt=1/30초로 10회 예측하여 미래 odom 값을 반환
  std::vector<std::pair<double, double>> predict_future_positions(double current_x, double current_y,
                                                     const std::vector<std::pair<double, double>> &positions,
                                                     KalmanFilter &kf, bool &kalman_initialized)
  {
    // 최근접 이웃 인덱스 계산 및 ±5 범위의 점 추출
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
    // 현재 측정값 포함
    sum_x += current_x;
    sum_y += current_y;
    count++;

    // 평균 측정값 계산
    double avg_x = sum_x / count;
    double avg_y = sum_y / count;
    RCLCPP_INFO(this->get_logger(), "Kalman measurement for %s prediction: avg_x=%.2f, avg_y=%.2f",
                "ego", avg_x, avg_y);  // (예시: ego)

    // 측정값 벡터 (2x1)
    Eigen::VectorXd y_meas(2);
    y_meas << avg_x, avg_y;

    // 칼만 필터 초기화 (미초기화 시)
    if (!kalman_initialized) {
      Eigen::VectorXd x0(4);
      x0 << avg_x, avg_y, 0, 0;
      kf.init(0.0, x0);
      kalman_initialized = true;
    } else {
      try {
        kf.update(y_meas);
      } catch (std::runtime_error &e) {
        RCLCPP_ERROR(this->get_logger(), "Kalman filter update error: %s", e.what());
      }
    }

    // 10회의 미래 예측 (dt=1/30초)
    std::vector<std::pair<double, double>> future_predictions;
    // getState() 메서드를 통해 현재 상태 벡터를 얻는다고 가정합니다.
    Eigen::VectorXd state = kf.getState();
    for (int i = 0; i < 10; i++) {
      state = A_kalman * state;  // 상태 예측: x_k+1 = A * x_k
      future_predictions.push_back({state(0), state(1)});
    }
    return future_predictions;
  }

  // --- 충돌/분리 여부 판단 함수 ---
  // (이전 코드와 동일)
  void check_collision(const std::vector<std::pair<double, double>> &predicted1,
                       const std::vector<std::pair<double, double>> &predicted2)
  {
    // 기존 변수 클리어
    colliding_predicted_points.clear();
    separating_predicted_points.clear();

    size_t steps = std::min(predicted1.size(), predicted2.size());
    for (size_t i = 0; i < steps; i++) {
      double dist = compute_distance(predicted1[i], predicted2[i]);
      // 충돌 예측 (거리 < 0.005)
      if (dist < 0.005) {
        colliding_predicted_points.push_back({predicted1[i], predicted2[i]});
      }
      // 분리 예측 (거리 >= 0.2)
      else if (dist >= 0.2) {
        separating_predicted_points.push_back({predicted1[i], predicted2[i]});
      }
    }
    if (!colliding_predicted_points.empty()) {
      RCLCPP_WARN(this->get_logger(), "Predicted colliding points: %ld pairs", colliding_predicted_points.size());
    }
    if (!separating_predicted_points.empty()) {
      RCLCPP_INFO(this->get_logger(), "Predicted separating points: %ld pairs", separating_predicted_points.size());
    }
  }

  // --- 예측 결과 시각화 함수 ---
  // ego odom의 미래 예측 점들을 Marker 메시지로 만들어 publish합니다.
  void publish_predicted_marker(const std::vector<std::pair<double, double>> &predicted_points)
  {
    visualization_msgs::msg::Marker marker;
    marker.header.frame_id = "odom";  // odom frame 기준 (필요에 따라 수정)
    marker.header.stamp = this->now();
    marker.ns = "predicted_points";
    marker.id = 0;
    marker.type = visualization_msgs::msg::Marker::SPHERE_LIST;
    marker.action = visualization_msgs::msg::Marker::ADD;
    marker.pose.orientation.w = 1.0;

    // Marker의 크기 설정 (각 구체의 직경)
    marker.scale.x = 0.05;
    marker.scale.y = 0.05;
    marker.scale.z = 0.05;

    // Marker 색상 (녹색)
    marker.color.r = 0.0f;
    marker.color.g = 1.0f;
    marker.color.b = 0.0f;
    marker.color.a = 1.0f;

    // 예측된 각 점을 geometry_msgs::msg::Point로 추가
    for (const auto &pt_pair : predicted_points) {
      geometry_msgs::msg::Point pt;
      pt.x = pt_pair.first;
      pt.y = pt_pair.second;
      pt.z = 0.0;  // 2D 평면 상
      marker.points.push_back(pt);
    }
    marker_pub->publish(marker);
  }

  // --- 유틸리티 함수들 ---
  // 두 점 사이의 유클리디안 거리 계산
  double compute_distance(const std::pair<double, double>& p1,
                          const std::pair<double, double>& p2)
  {
    double dx = p1.first - p2.first;
    double dy = p1.second - p2.second;
    return std::sqrt(dx*dx + dy*dy);
  }

  // 주어진 위치와 positions 내에서 가장 가까운 점의 인덱스 및 거리를 계산
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
  // 각 토픽 별 구독자
  rclcpp::Subscription<nav_msgs::msg::Odometry>::SharedPtr ego_subscription;
  rclcpp::Subscription<nav_msgs::msg::Odometry>::SharedPtr opponent_subscription;

  // RViz 시각화를 위한 Marker publisher
  rclcpp::Publisher<visualization_msgs::msg::Marker>::SharedPtr marker_pub;

  // odom 데이터 저장용 변수들
  std::vector<std::pair<double, double>> ego_odom_positions;
  std::vector<std::pair<double, double>> opponent_odom_positions;
  std::pair<double, double> ego_starting_odom;
  std::pair<double, double> opponent_starting_odom;
  bool ego_starting_initialized;
  bool ego_lap_completed;
  bool opponent_starting_initialized;
  bool opponent_lap_completed;

  // 공통 변수
  double distance_threshold;
  size_t min_message_before_check;
  double collision_threshold_dist; // 새로 추가한 멤버 변수 (0.02)

  // 칼만 필터 관련 변수 (constant velocity 모델)
  double dt_kalman;
  Eigen::MatrixXd A_kalman;
  Eigen::MatrixXd C_kalman;
  Eigen::MatrixXd Q_kalman;
  Eigen::MatrixXd R_kalman;
  Eigen::MatrixXd P0_kalman;

  // ego와 opponent 별 칼만 필터 인스턴스
  KalmanFilter kalman_filter_ego;
  KalmanFilter kalman_filter_opp;
  bool kalman_initialized_ego;
  bool kalman_initialized_opp;

  // 미래 예측 결과 저장 (각 10개씩)
  std::vector<std::pair<double, double>> predicted_future_ego;
  std::vector<std::pair<double, double>> predicted_future_opponent;

  // (이전 코드의 충돌/분리 예측 관련 변수)
  std::vector<std::pair<std::pair<double,double>, std::pair<double,double>>> colliding_predicted_points;
  std::vector<std::pair<std::pair<double,double>, std::pair<double,double>>> separating_predicted_points;
};

int main(int argc, char * argv[])
{
  rclcpp::init(argc, argv);
  auto node = std::make_shared<TrajectoryEstimation>();
  rclcpp::spin(node);
  rclcpp::shutdown();
  return 0;
}
