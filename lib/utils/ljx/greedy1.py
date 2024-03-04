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
        self.device = -1  # 设备index


def initial_selection_algorithm_greedy1(devices, session):
    global sumleft
    # 用户需要根据自己的逻辑来实现设备选择算法
    # 保证剩余的带宽大于会话需要的
    # 先剩余带宽排序
    # 直接返回最大的那个位置
    if sumleft < session.max_bandwidth:
        return False
    sorted_devices = sorted(devices, key=lambda d: d.max_bandwidth - d.current_SessionsMB, reverse=True)
    session.device = sorted_devices[0].id - 1
    devices[sorted_devices[0].id - 1].reserve_B(session.max_bandwidth, session)
    sumleft -= session.max_bandwidth
    print("deviceIndex: ", sorted_devices[0].id - 1)
    return True


def reSelection_algorithm_greedy1(devices, session):
    # 重新选择一个新的设备
    # 先剩余带宽排序
    # 直接返回最大的那个位置,如果超过大小就不分配了
    # 先恢复
    global sumleft
    orginalDev = session.device
    devices[orginalDev].current_SessionsMB -= session.max_bandwidth
    devices[orginalDev].current_bandwidth -= session.bandwidth

    sorted_devices = sorted(devices, key=lambda d: d.max_bandwidth - d.current_SessionsMB, reverse=True)

    if orginalDev != (sorted_devices[0].id - 1) and (sorted_devices[0].max_bandwidth - sorted_devices[0].current_bandwidth) > session.current_bandwidth:
        index = sorted_devices[0].id - 1
        devices[index].reserve_B(session.max_bandwidth, session)
        session.device = index
        return True
    else:
        devices[orginalDev].current_SessionsMB += session.max_bandwidth
        devices[orginalDev].current_bandwidth += session.bandwidth
        False


def update_bandwidth(scheduleNumber):
    for session in sessions:
        if session.bandwidth < session.max_bandwidth:
            additional_bandwidth = min(1, session.max_bandwidth - session.bandwidth)
            device = devices[session.device]
            if device.current_bandwidth + additional_bandwidth > device.max_bandwidth:
                # 存在过载的情况
                # 选择会话切换,添加选择更新为下一个

                if reSelection_algorithm_greedy1(devices, session):
                    # 成功找一个新的可以放置的device
                    # 迁移流去上面
                    scheduleNumber += session.current_bandwidth + additional_bandwidth
                    session.device.current_bandwidth += additional_bandwidth
                    session.bandwidth += additional_bandwidth
                else:
                    continue
            else:
                session.bandwidth += additional_bandwidth
                device.current_bandwidth += additional_bandwidth
                print("id: device.current_bandwidth:", device.id - 1, device.current_bandwidth)


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

sumleft = 0
scheduleNumber = 0

for i in range(1, deviceNum + 1):
    devices.append(Device(i, B))

for device in devices:
    sumleft += device.max_bandwidth - device.current_SessionsMB

timeEnd = 20
# 模拟每秒创建一个会话并更新带宽使用情况
for i in range(timeEnd):  # 假设模拟10秒
    new_session = Session(b, SessionId)
    SessionId += 1
    if initial_selection_algorithm_greedy1(devices, new_session):
        sessions.append(new_session)

    # 更新每个会话上的带宽
    for session in sessions:
        update_bandwidth(scheduleNumber)
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

print("system schedule times Number: ", scheduleNumber)
