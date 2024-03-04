class Device:
    def __init__(self, id, max_bandwidth):
        self.id = id
        self.max_bandwidth = max_bandwidth
        self.current_SessionsMB = 0
        self.current_bandwidth = 0
        self.sessions = []
        self.usage_rate = []

    def reserve_B(self, reserve, session):
        self.sessions.append(session)
        self.current_SessionsMB += reserve
        return True


class Session:
    def __init__(self, max_bandwidth, sessionId):
        self.id = sessionId
        self.max_bandwidth = max_bandwidth
        self.bandwidth = 0  # 初始带宽需求设置为0
        self.devices = []  # 设备index
        self.devices_reserve = []  # 设备预留资源
        self.currentDevice = -1  # 最后一次增加流的时候在什么设备上


def initial_selection_algorithm_greedy1(devices, session):
    # 用户需要根据自己的逻辑来实现设备选择算法
    # 保证剩余的带宽大于需要的
    # 先剩余带宽排序
    # 直接返回最大的那个位置
    sumleft = 0
    for device in devices:
        sumleft += device.max_bandwidth - device.current_SessionsMB
    # print("sumleft: ", sumleft)
    if sumleft < session.max_bandwidth:
        return False
    sorted_devices = sorted(devices, key=lambda d: d.max_bandwidth - d.current_SessionsMB, reverse=True)
    needB = session.max_bandwidth
    while True:
        if needB > 0:
            index = sorted_devices[0].id - 1
            reserve = min(needB, devices[index].max_bandwidth - devices[index].current_SessionsMB)
            needB -= reserve
            if len(session.devices) == 0:
                session.currentDevice = index
            # 更新设备预留
            devices[index].reserve_B(reserve, session)
            session.devices.append(index)
            session.devices_reserve.append(reserve)
            sorted_devices = sorted(devices, key=lambda d: d.max_bandwidth - d.current_SessionsMB, reverse=True)
        else:
            break
    return True


def update_bandwidth():
    for session in sessions:
        if session.bandwidth < session.max_bandwidth:
            additional_bandwidth = min(1, session.max_bandwidth - session.bandwidth)
            device = devices[session.currentDevice]
            if device.current_bandwidth + additional_bandwidth > device.max_bandwidth:
                # 存在过载的情况
                # 选择流切换,添加选择更新为下一个
                for i in range(0, len(session.devices)):
                    if session.devices[i] == (device.id - 1):
                        session.currentDevice = session.devices[i + 1]
                device = devices[session.currentDevice]
            # 一定找到了一台设备
            session.bandwidth += additional_bandwidth
            device.current_bandwidth += additional_bandwidth
            print("device_id: %d, device.current_bandwidth: %d" % (device.id, device.current_bandwidth))


def update_device_bandwidth_usage(device):
    device.usage_rate.append(device.current_bandwidth / device.max_bandwidth)


# 设定设备和会话的带宽限制
B = 10  # 设备的带宽限制
b = 4  # 会话的带宽限制
SessionId = 0
# 创建3个带宽限制为B的设备
devices = []
deviceNum = 3
sessions = []

for i in range(1, deviceNum + 1):
    devices.append(Device(i, B))

timeEnd = 20
# 模拟每秒创建一个会话并更新带宽使用情况
for i in range(timeEnd):  # 假设模拟10秒
    new_session = Session(b, SessionId)
    SessionId += 1
    if initial_selection_algorithm_greedy1(devices, new_session):
        sessions.append(new_session)

    # 更新每个会话上的带宽
    for session in sessions:
        update_bandwidth()
    # 更新每个设备上的带宽利用
    for device in devices:
        update_device_bandwidth_usage(device)

devicesBSum = 0

for device in devices:
    devicesBSum += device.max_bandwidth

devicesRate = []
for i in range(timeEnd):
    rate = 0
    for device in devices:
        rate += device.usage_rate[i] * device.max_bandwidth / devicesBSum
    devicesRate.append(rate)

print("System UsageRate with Time: ", devicesRate)

scheduleNumber = 0
for session in sessions:
    scheduleNumber += len(session.devices) - 1

print("system schedule times Number: ", scheduleNumber)
