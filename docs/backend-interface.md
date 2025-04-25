# 轨迹表(Trajectory)接口文档

## 基本信息
- 基础路径: `/api/trajectory`
- 数据类型: JSON

## API端点

### 1. 获取所有轨迹数据

- **URL**: `/api/trajectory`
- **方法**: GET
- **响应**: 轨迹数据数组
```json
[
  {
    "id": 1,
    "deviceId": "radar001",
    "targetId": "person001",
    "timestamp": "2023-04-20T14:30:00",
    "position": {"x": 1.2, "y": 3.4, "z": 0.5},
    "pointclouds": [{"x": 1.1, "y": 3.3, "z": 0.4}, ...],
    "direction": 45.5,
    "createdAt": "2023-04-20T14:30:05"
  },
  ...
]
```

### 2. 按设备ID获取轨迹数据

- **URL**: `/api/trajectory/device/{deviceId}`
- **方法**: GET
- **参数**: 
  - `deviceId`: 设备ID
- **响应**: 轨迹数据数组(按时间戳降序)

### 3. 添加轨迹数据

- **URL**: `/api/trajectory`
- **方法**: POST
- **请求体**:
```json
{
  "deviceId": "radar001",
  "targetId": "person001",
  "timestamp": "2023-04-20T14:30:00",
  "position": {"x": 1.2, "y": 3.4, "z": 0.5},
  "pointclouds": [{"x": 1.1, "y": 3.3, "z": 0.4}, ...],
  "direction": 45.5
}
```
- **响应**: 创建的轨迹数据对象

### 4. 更新轨迹数据

- **URL**: `/api/trajectory/{id}`
- **方法**: PUT
- **参数**: 
  - `id`: 轨迹数据ID
- **请求体**: 与添加接口相同
- **响应**: 更新后的轨迹数据对象

### 5. 删除轨迹数据

- **URL**: `/api/trajectory/{id}`
- **方法**: DELETE
- **参数**: 
  - `id`: 轨迹数据ID
- **响应**: 
  - 成功: HTTP 200
  - 未找到: HTTP 404

## 数据结构说明

| 字段名      | 类型     | 说明                   |
| ----------- | -------- | ---------------------- |
| id          | Long     | 主键ID                 |
| deviceId    | String   | 雷达设备ID             |
| targetId    | String   | 目标ID                 |
| timestamp   | String   | 时间戳，格式：ISO-8601 |
| position    | JSON对象 | 位置坐标数据           |
| pointclouds | JSON数组 | 点云数据               |
| direction   | Float    | 移动方向(度)           |
| createdAt   | String   | 记录创建时间           |