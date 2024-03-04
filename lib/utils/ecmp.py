class Device:
    def __init__(self, id, max_bandwidth):
        self.id = id
        self.max_bandwidth = max_bandwidth
        self.current_SessionsMB = 0
        self.current_bandwidth = 0
        self.sessions = []
        self.usage_rate = []

    def add_session(self, session):
        # 只有在不超过最大带宽时才添加会话
        if self.max_bandwidth - self.current_SessionsMB > session.max_bandwidth:
            self.sessions.append(session)
            self.current_SessionsMB += session.max_bandwidth
            self.current_bandwidth += session.bandwidth
            return True
        return False

    def update_bandwidth(self, selected_device_id: int):
        # 为每个会话增加带宽需求，直到会话达到其最大带宽限制
        for session in self.sessions:
            if session.bandwidth < session.max_bandwidth and self.id == selected_device_id:
                additional_bandwidth = min(1, session.max_bandwidth - session.bandwidth)
                if self.current_bandwidth + additional_bandwidth > self.max_bandwidth:
                    break  # 如果增加后超过设备带宽，则不再增加
                session.bandwidth += additional_bandwidth
                self.current_bandwidth += additional_bandwidth
        self.usage_rate.append(self.current_bandwidth / self.max_bandwidth)


class Session:
    def __init__(self, max_bandwidth, sessionId):
        self.id = sessionId
        self.max_bandwidth = max_bandwidth
        self.bandwidth = 0  # 初始带宽需求设置为1


def initial_selection_algorithm_ecmp(devices, session):
    # 用户需要根据自己的逻辑来实现设备选择算法
    # 直接哈希分配上去
    # 会话带宽不满足需求就不分配
    return devices[session.id % len(devices)]
    # if  devices[deviceId].max_bandwidth - devices[deviceId].current_SessionsMB > session.max_bandwidth:
    #    return devices[deviceId]


# 设定设备和会话的带宽限制
B = 10  # 设备的带宽限制
b = 4  # 会话的带宽限制
SessionId = 0
# 创建3个带宽限制为B的设备
devices = []
deviceNum = 3
for i in range(1, deviceNum + 1):
    devices.append(Device(i, B))

timeEnd = 20
# 模拟每秒创建一个会话并更新带宽使用情况
for i in range(timeEnd):  # 假设模拟10秒
    new_session = Session(b, SessionId)  # 创建新会话
    SessionId += 1
    selected_device = initial_selection_algorithm_ecmp(devices, new_session)
    if selected_device:
        selected_device.add_session(new_session)

    # 更新每个设备上的带宽使用情况
    for device in devices:
        device.update_bandwidth(selected_device.id)

    # 打印每个设备上的当前总带宽使用情况

devicesBSum = 0

for device in devices:
    devicesBSum += device.max_bandwidth

devicesRate = []
for i in range(timeEnd):
    rate = 0
    for device in devices:
        rate += device.usage_rate[i] * device.max_bandwidth / devicesBSum
    devicesRate.append(rate)

print("System UsageRate with Time: ")
print(devicesRate)
