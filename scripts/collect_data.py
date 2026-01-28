#!/usr/bin/env python3
import rospy
import csv
import math
import time
import random
import numpy as np
from geometry_msgs.msg import PoseStamped, TwistStamped
from sensor_msgs.msg import Imu
from gazebo_msgs.msg import LinkStates
from mavros_msgs.msg import State, AttitudeTarget # [新增] 引入 AttitudeTarget
from mavros_msgs.srv import CommandBool, SetMode

# --- 配置部分 ---
CSV_FILE_PATH = "./uav_slung_load/slung_load_dataset.csv"
RECORD_RATE = 50

# Gazebo Link 名称
DRONE_LINK_NAME = 'iris::base_link'     
LOAD_LINK_NAME = 'iris::payload'        

class DataCollector:
    def __init__(self):
        self.csv_file = open(CSV_FILE_PATH, 'w', newline='')
        self.writer = csv.writer(self.csv_file)
        
        # 表头 [新增 cmd_thrust]
        self.header = [
            'timestamp', 
            'imu_ax', 'imu_ay', 'imu_az', 'imu_wx', 'imu_wy', 'imu_wz',
            'drone_px', 'drone_py', 'drone_pz', 'drone_vx', 'drone_vy', 'drone_vz',
            'drone_qw', 'drone_qx', 'drone_qy', 'drone_qz', 
            'cmd_thrust',  # 控制器输出的推力 (0~1)
            'gt_xi', 'gt_zeta', 'gt_xi_dot', 'gt_zeta_dot',
            'gt_load_px', 'gt_load_py', 'gt_load_pz'
        ]
        self.writer.writerow(self.header)

        self.current_imu = None
        self.current_pose = None
        self.current_vel = None
        self.current_cmd = None # [新增] 用于存储控制指令
        
        self.drone_gt_pose = None
        self.load_gt_pose = None
        
        self.link_name_debug_printed = False

        # 差分变量
        self.last_xi = 0.0
        self.last_zeta = 0.0
        self.last_time = None 

        rospy.Subscriber('/mavros/imu/data', Imu, self.imu_cb)
        rospy.Subscriber('/mavros/local_position/pose', PoseStamped, self.pose_cb)
        rospy.Subscriber('/mavros/local_position/velocity_local', TwistStamped, self.vel_cb)
        rospy.Subscriber('/gazebo/link_states', LinkStates, self.gazebo_cb)
        
        # [关键新增] 订阅飞控的目标姿态与推力
        # 注意：当你发送位置 Setpoint 时，PX4 内部的位置控制器会算出 AttitudeTarget
        # 这个 Topic 通常是 /mavros/setpoint_raw/target_attitude
        rospy.Subscriber('/mavros/setpoint_raw/target_attitude', AttitudeTarget, self.cmd_cb)

    def imu_cb(self, msg): self.current_imu = msg
    def pose_cb(self, msg): self.current_pose = msg
    def vel_cb(self, msg): self.current_vel = msg
    
    # 记录控制指令的回调
    def cmd_cb(self, msg): self.current_cmd = msg

    def gazebo_cb(self, msg):
        try:
            drone_idx = msg.name.index(DRONE_LINK_NAME)
            load_idx = msg.name.index(LOAD_LINK_NAME)
            self.drone_gt_pose = msg.pose[drone_idx]
            self.load_gt_pose = msg.pose[load_idx]
            
            if not self.link_name_debug_printed:
                print(f"[DataCollector] Successfully found links: {DRONE_LINK_NAME}, {LOAD_LINK_NAME}")
                self.link_name_debug_printed = True
        except ValueError:
            if not self.link_name_debug_printed:
                print(f"\n[ERROR] Link Name Mismatch! Waiting for: '{DRONE_LINK_NAME}'...")
                self.link_name_debug_printed = True
            pass

    def calculate_swing_angles(self):
        if self.drone_gt_pose is None or self.load_gt_pose is None:
            return 0, 0, 0, 0
            
        dp = self.drone_gt_pose.position
        lp = self.load_gt_pose.position
        dx, dy, dz = lp.x - dp.x, lp.y - dp.y, lp.z - dp.z
        
        # 1. 计算角度
        xi = math.atan2(dx, -dz) 
        zeta = math.atan2(dy, -dz)
        
        # 2. 计算角速度
        now = rospy.Time.now().to_sec()
        xi_dot = 0.0
        zeta_dot = 0.0
        
        if self.last_time is not None:
            dt = now - self.last_time
            if dt > 0.0001 and dt < 1.0:
                xi_dot = (xi - self.last_xi) / dt
                zeta_dot = (zeta - self.last_zeta) / dt
            
        self.last_xi = xi
        self.last_zeta = zeta
        self.last_time = now
        
        return xi, zeta, xi_dot, zeta_dot

    def record_step(self, event):
        # 必须确保核心数据都有了才记录
        if self.csv_file.closed:
            return
            
        if self.current_imu is None or self.drone_gt_pose is None or self.current_pose is None or self.current_vel is None:
            return
        
        # 如果还没收到推力指令，暂时记为 0 (通常起飞前会是 0)
        thrust_val = 0.0
        if self.current_cmd is not None:
            thrust_val = self.current_cmd.thrust

        xi, zeta, xi_dot, zeta_dot = self.calculate_swing_angles()
        q = self.current_pose.pose.orientation 

        row = [
            rospy.Time.now().to_sec(),
            self.current_imu.linear_acceleration.x, self.current_imu.linear_acceleration.y, self.current_imu.linear_acceleration.z,
            self.current_imu.angular_velocity.x, self.current_imu.angular_velocity.y, self.current_imu.angular_velocity.z,
            self.current_pose.pose.position.x, self.current_pose.pose.position.y, self.current_pose.pose.position.z,
            self.current_vel.twist.linear.x, self.current_vel.twist.linear.y, self.current_vel.twist.linear.z,
            q.w, q.x, q.y, q.z,
            thrust_val, # [新增] 这里的 thrust 是 0~1 的归一化值
            xi, zeta, xi_dot, zeta_dot,
            self.load_gt_pose.position.x, self.load_gt_pose.position.y, self.load_gt_pose.position.z
        ]
        
        if not self.csv_file.closed:
            try:
                self.writer.writerow(row)
            except ValueError:
                pass

    def close(self):
        if not self.csv_file.closed:
            self.csv_file.close()

class Commander:
    def __init__(self):
        self.pose_pub = rospy.Publisher('/mavros/setpoint_position/local', PoseStamped, queue_size=10)
        self.current_state = State()
        self.current_pose = PoseStamped()
        
        rospy.Subscriber('/mavros/state', State, self.state_cb)
        rospy.Subscriber('/mavros/local_position/pose', PoseStamped, self.pose_cb)

    def state_cb(self, msg): self.current_state = msg
    def pose_cb(self, msg): self.current_pose = msg

    def publish_setpoint(self, x, y, z):
        pose = PoseStamped()
        pose.header.stamp = rospy.Time.now()
        pose.pose.position.x = x
        pose.pose.position.y = y
        pose.pose.position.z = z
        pose.pose.orientation.w = 1.0 
        self.pose_pub.publish(pose)

    def ensure_offboard_takeoff(self, target_height=5.0):
        rate = rospy.Rate(20)
        print("Waiting for MAVROS connection...")
        while not rospy.is_shutdown() and not self.current_state.connected:
            rate.sleep()
        print("Connected!")

        start_x = self.current_pose.pose.position.x
        start_y = self.current_pose.pose.position.y
        current_z = self.current_pose.pose.position.z
        target_z = max(target_height, current_z) 
        
        print(f"Sending setpoints... Target Z: {target_z:.2f}")
        for _ in range(50):
            if rospy.is_shutdown(): break
            self.publish_setpoint(start_x, start_y, target_z)
            rate.sleep()

        last_req = rospy.Time.now()
        
        while not rospy.is_shutdown():
            self.publish_setpoint(start_x, start_y, target_z)

            now = rospy.Time.now()
            if now - last_req > rospy.Duration(5.0):
                last_req = now
                
                if not self.current_state.armed and self.current_pose.pose.position.z < 0.5:
                    try:
                        arming_cl = rospy.ServiceProxy('/mavros/cmd/arming', CommandBool)
                        resp = arming_cl(True)
                        if resp.success: print("Vehicle ARMED")
                    except rospy.ServiceException as e:
                        print(f"Arming failed: {e}")
                
                if self.current_state.mode != "OFFBOARD":
                    try:
                        set_mode_cl = rospy.ServiceProxy('/mavros/set_mode', SetMode)
                        resp = set_mode_cl(custom_mode='OFFBOARD')
                        if resp.mode_sent: 
                            print("Offboard enabled")
                    except rospy.ServiceException as e:
                        print(f"Set mode failed: {e}")

            if self.current_state.mode == "OFFBOARD" and self.current_state.armed:
                if abs(self.current_pose.pose.position.z - target_z) < 0.3:
                    print("Takeoff/Hover Stable. Ready for Trajectory.")
                    break
            
            rate.sleep()

    def run_trajectory(self):
        print("Starting [VERY GENTLE] Trajectory Collection (Target: 5 mins)...")
        rate = rospy.Rate(30)
        
        start_time = time.time()
        TOTAL_DURATION = 3 * 60  # 5 minutes
        
        def wait_and_pub(duration, x, y, z):
            end_t = time.time() + duration
            while time.time() < end_t and not rospy.is_shutdown():
                self.publish_setpoint(x, y, z)
                rate.sleep()

        current_phase_idx = 0
        
        # Gentle pattern list
        patterns = ['lissajous', 'circle', 'random_walk', 'spiral', 'wave', 'stop_and_go', 'hover_loiter']
        
        while (time.time() - start_time < TOTAL_DURATION) and not rospy.is_shutdown():
            # Randomly select a pattern
            phase_type = random.choice(patterns)
            duration = random.uniform(20, 40) 
            phase_start = time.time()
            
            current_phase_idx += 1
            print(f"\n--- Phase {current_phase_idx}: {phase_type.upper()} (Duration: {duration:.1f}s) ---")
            print(f"Time Remaining: {(TOTAL_DURATION - (time.time() - start_time))/60:.1f} mins")
            
            if phase_type == 'lissajous':
                # [修改点 1] 减小利萨茹曲线的频率 (a, b) 和幅度 (A, B)
                A = random.uniform(1.5, 3.0)  # 原: 2-5 -> 现: 1.5-3.0
                B = random.uniform(1.5, 3.0)
                a = random.uniform(0.1, 0.25) # 原: 0.2-0.5 -> 现: 0.1-0.25 (慢一倍)
                b = random.uniform(0.1, 0.25)
                delta = random.uniform(0, math.pi)
                z_base = random.uniform(4, 8)
                print(f"Params: A={A:.1f}, B={B:.1f}, a={a:.2f}, b={b:.2f}")
                
                while (time.time() - phase_start < duration) and not rospy.is_shutdown():
                    t = time.time() - phase_start
                    x = A * math.sin(a * t + delta)
                    y = B * math.sin(b * t)
                    z = z_base + 0.3 * math.sin(0.1 * t) 
                    self.publish_setpoint(x, y, z)
                    rate.sleep()
                    
            elif phase_type == 'circle':
                # [修改点 2] 大幅减小圆周运动的速度
                # 之前的 1.5 rad/s 在 R=5m 时向心加速度极大，是产生大摆角的主因
                R = random.uniform(2, 4)      # 原: 2-6 -> 现: 2-4 (减小半径)
                speed = random.uniform(0.3, 0.6) * random.choice([1, -1]) # 原: 0.8-1.5 -> 现: 0.3-0.6 (非常慢)
                z_base = random.uniform(4, 8)
                print(f"Params: R={R:.1f}, Speed={speed:.2f}")
                
                while (time.time() - phase_start < duration) and not rospy.is_shutdown():
                    t = time.time() - phase_start
                    x = R * math.cos(speed * t)
                    y = R * math.sin(speed * t)
                    z = z_base
                    self.publish_setpoint(x, y, z)
                    rate.sleep()

            elif phase_type == 'hover_loiter':
                # 悬停本来就很稳，稍微减小漂移幅度即可
                tx = random.uniform(-4, 4)
                ty = random.uniform(-4, 4)
                tz = random.uniform(3, 8)
                print(f"Loitering at ({tx:.1f}, {ty:.1f}, {tz:.1f})")
                
                drift_amp = 0.3 # 原 0.5
                drift_freq = 0.1 # 原 0.2
                
                while (time.time() - phase_start < duration) and not rospy.is_shutdown():
                    t = time.time() - phase_start
                    x = tx + drift_amp * math.sin(drift_freq * t)
                    y = ty + drift_amp * math.cos(drift_freq * t * 0.7)
                    self.publish_setpoint(x, y, tz)
                    rate.sleep()
                    
            elif phase_type == 'spiral':
                # [修改点 3] 减小螺旋上升/下降的速度
                R = random.uniform(2, 4)
                speed = random.uniform(0.2, 0.5) # 原: 0.3-0.8
                z_start = random.uniform(3, 8)
                z_speed = random.choice([0.15, -0.15]) # 原: 0.2
                print(f"Params: R={R:.1f}, Speed={speed:.2f}, Z_speed={z_speed}")
                
                while (time.time() - phase_start < duration) and not rospy.is_shutdown():
                    t = time.time() - phase_start
                    x = R * math.cos(speed * t)
                    y = R * math.sin(speed * t)
                    z_curr = z_start + z_speed * t
                    
                    if z_curr < 2.0: 
                        z_curr = 2.0
                        z_speed = abs(z_speed)
                        z_start = z_curr - z_speed * t 
                    elif z_curr > 10.0:
                        z_curr = 10.0
                        z_speed = -abs(z_speed)
                        z_start = z_curr - z_speed * t
                        
                    self.publish_setpoint(x, y, z_curr)
                    rate.sleep()

            elif phase_type == 'wave':
                # [修改点 4] 减小波浪飞行的频率
                L = random.uniform(8, 12)
                freq = random.uniform(0.2, 0.4) # 原: 0.3-0.8 -> 降低频率
                amp = random.uniform(1, 2)
                z_base = random.uniform(4, 8)
                print(f"Params: Wave Length={L:.1f}, Freq={freq:.2f}, Amp={amp:.1f}")
                
                while (time.time() - phase_start < duration) and not rospy.is_shutdown():
                    t = time.time() - phase_start
                    cycle_time = 30.0 # 增加周期时间让直线段更平缓
                    norm_t = (t % cycle_time) / cycle_time 
                    if norm_t < 0.5:
                         curr_x = -L/2 + (L * (norm_t * 2))
                    else:
                         curr_x = L/2 - (L * ((norm_t - 0.5) * 2))
                    
                    curr_y = amp * math.sin(freq * t)
                    self.publish_setpoint(curr_x, curr_y, z_base)
                    rate.sleep()

            elif phase_type == 'stop_and_go':
                # [修改点 5] 缩小目标点范围，避免长距离急加速
                print("Performing Gentle Stop-and-Go...")
                while (time.time() - phase_start < duration) and not rospy.is_shutdown():
                    tx = random.uniform(-3, 3) # 原: -6, 6 -> 缩小范围
                    ty = random.uniform(-3, 3)
                    tz = random.uniform(3, 8)
                    
                    # 增加等待时间让负载摆动衰减
                    wait_and_pub(random.uniform(5.0, 7.0), tx, ty, tz)
                    wait_and_pub(random.uniform(5.0, 7.0), tx, ty, tz)

            elif phase_type == 'step':
                print("executing small steps...")
                while (time.time() - phase_start < duration) and not rospy.is_shutdown():
                    tx = random.uniform(-3, 3) # 缩小范围
                    ty = random.uniform(-3, 3)
                    tz = random.uniform(4, 8)
                    step_hold = random.uniform(5, 8) # 增加保持时间
                    wait_and_pub(step_hold, tx, ty, tz)
                    
            elif phase_type == 'random_walk':
                # [修改点 6] 随机漫游也缩小范围并增加间隔
                tx, ty, tz = 0, 0, 5
                next_change = 0
                print("executing gentle random walk...")
                while (time.time() - phase_start < duration) and not rospy.is_shutdown():
                    now = time.time()
                    if now > next_change:
                        tx = random.uniform(-4, 4) # 原: -7, 7
                        ty = random.uniform(-4, 4)
                        tz = random.uniform(3, 8)
                        next_change = now + random.uniform(6.0, 10.0) # 增加间隔
                    self.publish_setpoint(tx, ty, tz)
                    rate.sleep()
        
        print("Collection Finished.")
        print("Returning Home...")
        wait_and_pub(8.0, 0, 0, 5)
        print("Done.")

if __name__ == '__main__':
    rospy.init_node('slung_load_data_collector')
    
    collector = DataCollector()
    # 保持 50Hz 采样
    rospy.Timer(rospy.Duration(1.0/RECORD_RATE), collector.record_step)
    
    commander = Commander()
    try:
        commander.ensure_offboard_takeoff()
        commander.run_trajectory()
    except rospy.ROSInterruptException:
        pass
    finally:
        collector.close()
        print("Mission Finished.")