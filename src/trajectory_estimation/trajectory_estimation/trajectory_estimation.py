import rclpy
from rclpy.node import Node
from nav_msgs.msg import Odometry
import math

class TrajectoryEstimation(Node):
    def __init__(self):
        super().__init__("Trajectory_Estimation")
        self.subscription = self.create_subscription(
            Odometry,
            "/odom", # 파라미터화 할 것
            self.odom_callback,
            10
        )
        self.subscription # 참조를 위해서 사용

        self.odom_data = [] # odometry 데이터를 저장하기 위해서 사용
        self.starting_odom = None # 가장 처음 시작할 odom
        self.lap_completed = False # loop closure 판단 플래그
        self.distance_threshold = 0.2 # Loop Clousre 판단을 위한 최소 거리
        # 파라미터화 할 것
        self.min_message_before_check = 10 # 최소한의 odom 판단을 위한 토픽 메시지 수
        # 파라미터화 할 것

    def compute_distance(self, odom1:Odometry, odom2:Odometry):
        # loop closing 판단 여부
        dx = odom1.pose.pose.position.x - odom2.pose.pose.position.x
        dy = odom1.pose.pose.position.y - odom2.pose.pose.position.y
        return math.sqrt(dx**2 + dy**2)

    def odom_callback(self, msg:Odometry):
        self.get_logger().info(
            f"Received odometry: position=({msg.pose.pose.position.x:.2f},{msg.pose.pose.position.y:.2f})")

        if not self.lap_completed:
            self.odom_data.append(msg)

            if self.starting_odom is None:
                self.starting_odom = msg
                return

            if len(self.odom_data) >= self.min_message_before_check:
                distance = self.compute_distance(msg, self.starting_odom)
                self.get_logger().info(f"Distance from start: {distance:.2f}")

                if distance < self.distance_threshold:
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

