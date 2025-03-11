import rclpy
from rclpy.node import Node
from nav_msgs.msg import Odometry
import math
import numpy as np

class TrajectoryEstimation(Node):
    def __init__(self):
        super().__init__("Trajectory_Estimation")
        self.subscription = self.create_subscription(
            Odometry,
            "/odom",
            self.odom_callback,
            10
        )
        """odom 토픽 이름은 파라미터화 할 것 """
        self.subscription # 참조를 위해서 사용

        self.odom_positions = [] # odometry 데이터를 저장하기 위해서 사용
        self.starting_odom = None # 가장 처음 시작할 odom
        self.lap_completed = False # loop closure 판단 플래그
        self.distance_threshold = 0.2 # Loop Clousre 판단을 위한 최소 거리
        """ 파라미터화 할 것 """
        self.min_message_before_check = 10 # 최소한의 odom 판단을 위한 토픽 메시지 수
        """ 파라미터화 할 것 """
        self.odom_positions_np = None # 빠른 계산을 위해 Numpy 배열을 선언

    def compute_distance(self, odom1:tuple, odom2:tuple):
        # loop closing 판단 여부를 계산하기 위해서 작성한 함수
        # 튜플로 개선함
        dx = odom1[0] - odom2[0]
        dy = odom1[1] - odom2[1]
        return math.sqrt(dx**2 + dy**2)

    def get_nearest_neighbor_index(self, new_position):
        # 저장된 odom 배열과 새로 들어온 odom 데이터를 비교
        # 그중 신규 연산과 가장 가까운 점을 찾아서 인덱스를 반환
        new_pos = np.array(new_position) # numpy 배열로 변환
        diff = self.odom_positions_np - new_pos
        # 저장된 모든 위치와의 차이 계산
        # 각 차이의 norm을 계산
        distances = np.linalg.norm(diff, axis=1)
        min_index = np.argmin(distances)
        min_distance = distances[min_index]
        return min_index, min_distance

    def odom_callback(self, msg:Odometry):
        x = msg.pose.pose.position.x
        y = msg.pose.pose.position.y
        current_position = (x, y)
        self.get_logger().info(f"Received odometry: position=({x:.2f},{y:.2f})")

        if not self.lap_completed: # loop closure 판단이 되지 않았을 때만 실행
            self.odom_positions.append(current_position)

            if self.starting_odom is None: # 처음 시작할 odom 데이터를 저장
                self.starting_odom = current_position
                return

            if len(self.odom_positions) >= self.min_message_before_check:
                # 최소한의 odom 데이터가 들어왔을 때만 loop closure 판단
                distance = self.compute_distance(current_position, self.starting_odom)
                self.get_logger().info(f"Distance from start: {distance:.2f}")

                if distance < self.distance_threshold:
                    # 처음 시작한 odom 데이터와 현재 odom 데이터의 거리가
                    # threshold보다 작을 때 loop closure 판단
                    self.lap_completed = True
                    self.odom_positions_np = np.array(self.odom_positions)
                    self.get_logger().info(f"Lap completed! Stopping odom collection. Odom data size: {len(self.odom_positions)}")
        else:
            index, distance = self.get_nearest_neighbor_index(current_position)
            self.get_logger().info(f"Nearest neighbor index: {index}, distance: {distance:.2f}")

def main(args=None):
    rclpy.init(args=args)
    trajectory_estimation = TrajectoryEstimation()
    try:
        rclpy.spin(trajectory_estimation)
    except KeyboardInterrupt:
        trajectory_estimation.destroy_node()
    finally:
        rclpy.shutdown()

if __name__ == "__main__":
    main()

