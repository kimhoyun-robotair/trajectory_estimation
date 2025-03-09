import rclpy
from rclpy.node import Node
from nav_msgs.msg import Odometry
import math

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

        self.odom_data = [] # odometry 데이터를 저장하기 위해서 사용
        self.starting_odom = None # 가장 처음 시작할 odom
        self.lap_completed = False # loop closure 판단 플래그
        self.distance_threshold = 0.2 # Loop Clousre 판단을 위한 최소 거리
        """ 파라미터화 할 것 """
        self.min_message_before_check = 10 # 최소한의 odom 판단을 위한 토픽 메시지 수
        """ 파라미터화 할 것 """

    def compute_distance(self, odom1:Odometry, odom2:Odometry):
        # loop closing 판단 여부를 계산하기 위해서 작성한 함수
        dx = odom1.pose.pose.position.x - odom2.pose.pose.position.x
        dy = odom1.pose.pose.position.y - odom2.pose.pose.position.y
        return math.sqrt(dx**2 + dy**2)

    def get_nearest_neighbor_index(self, new_msg):
        # 저장된 odom 배열과 새로 들어온 odom 데이터를 비교
        # 그중 신규 연산과 가장 가까운 점을 찾아서 인덱스를 반환
        min_distance = float("inf") # 무한대
        """추후 파라미터화할 필요 있을 것 같음"""
        min_index = -1
        for i, stored_msg in enumerate(self.odom_data):
            distance = self.compute_distance(new_msg, stored_msg)
            if distance < min_distance:
                min_distance = distance
                min_index = i
        return min_index, min_distance

    def odom_callback(self, msg:Odometry):
        self.get_logger().info(
            f"Received odometry: position=({msg.pose.pose.position.x:.2f},{msg.pose.pose.position.y:.2f})")

        if not self.lap_completed: # loop closure 판단이 되지 않았을 때만 실행
            self.odom_data.append(msg)

            if self.starting_odom is None: # 처음 시작할 odom 데이터를 저장
                self.starting_odom = msg
                return

            if len(self.odom_data) >= self.min_message_before_check:
                # 최소한의 odom 데이터가 들어왔을 때만 loop closure 판단
                distance = self.compute_distance(msg, self.starting_odom)
                self.get_logger().info(f"Distance from start: {distance:.2f}")

                if distance < self.distance_threshold:
                    # 처음 시작한 odom 데이터와 현재 odom 데이터의 거리가
                    # threshold보다 작을 때 loop closure 판단
                    self.lap_completed = True
                    self.get_logger().info("Lap completed!")
        else:
            pass

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

