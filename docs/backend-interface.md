# 接口文档

---

## 1. 设备管理 (RadarDevice)

### 1.1 获取所有雷达设备 
- **URL**: `/api/radar/devices`
- **方法**: `GET`
- **描述**: 获取系统中所有注册的雷达设备列表
- **响应格式**: 
  ```json
  [
    {
      "deviceId": "string",      // 雷达设备唯一标识
      "deviceName": "string",    // 设备名称
      "model": "string",         // 设备型号
      "location": "string",      // 安装位置
      "type": "string",          // 类型
      "status": "string",        // 状态（在线/离线）
      "createdAt": "ISO时间戳",  // 创建时间
      "updatedAt": "ISO时间戳"   // 最后更新时间
    },
    ...
  ]
  ```

### 1.2 获取单个雷达设备
- **URL**: `/api/radar/devices/{deviceId}`
- **方法**: `GET`
- **路径参数**:
  - `deviceId`: 设备唯一标识 (String)
- **响应格式**: 
  ```json
  {
    "deviceId": "string",
    "deviceName": "string",
    "model": "string",
    "location": "string",
    "type": "string",
    "status": "string",
    "createdAt": "ISO时间戳",
    "updatedAt": "ISO时间戳"
  }
  ```
- **状态码**:
  - `200 OK`: 成功
  - `404 Not Found`: 设备不存在

### 1.3 添加新的雷达设备
- **URL**: `/api/radar/devices`
- **方法**: `POST`
- **请求体**:
  ```json
  {
    "deviceId": "string",      // 必需，设备唯一标识
    "deviceName": "string",    // 必需，设备名称
    "model": "string",         // 可选，设备型号
    "location": "string",      // 可选，安装位置
    "type": "string",          // 可选，类型
    "status": "string"         // 可选，状态
  }
  ```
- **响应**: 返回创建的设备对象，格式同上
- **状态码**:
  
  - `200 OK`: 成功创建
  - `400 Bad Request`: 请求参数错误

### 1.4 更新雷达设备
- **URL**: `/api/radar/devices/{deviceId}`
- **方法**: `PUT`
- **路径参数**:
  - `deviceId`: 设备唯一标识
- **请求体**: 同添加设备请求体
- **响应**: 返回更新后的设备对象
- **状态码**:
  - `200 OK`: 成功更新
  - `404 Not Found`: 设备不存在
  - `400 Bad Request`: 请求参数错误

### 1.5 删除雷达设备
- **URL**: `/api/radar/devices/{deviceId}`
- **方法**: `DELETE`
- **路径参数**:
  - `deviceId`: 设备唯一标识
- **响应**: 无响应体
- **状态码**:
  - `200 OK`: 成功删除
  - `404 Not Found`: 设备不存在

---

## 2. 生命体征 (VitalSigns)

### 2.1 获取所有生命体征数据
- **URL**: `/api/vital-signs`
- **方法**: `GET`
- **描述**: 获取所有生命体征数据记录
- **响应格式**: 
  ```json
  [
    {
      "id": long,                   // 记录ID
      "deviceId": "string",         // 设备ID
      "timestamp": "ISO时间戳",     
      "heartRate": "json字符串",    // 心率数据JSON
      "respirationRate": "json字符串", // 呼吸率数据JSON
      "bodyTemp": float,            // 体温(摄氏度)
      "movementLevel": float,       // 活动水平(0-1)
      "createdAt": "ISO时间戳"      // 创建时间
    },
    ...
  ]
  ```

### 2.2 获取指定设备的生命体征数据
- **URL**: `/api/vital-signs/device/{deviceId}`
- **方法**: `GET`
- **路径参数**:
  - `deviceId`: 设备唯一标识
- **响应格式**: 同上，返回符合条件的数据列表

### 2.3 添加新的生命体征数据
- **URL**: `/api/vital-signs`
- **方法**: `POST`
- **请求体**:
  ```json
  {
    "deviceId": "string",          // 必需，设备ID
    "timestamp": "ISO时间戳",      // 可选，默认为当前时间
    "heartRate": {                 // 可选，心率相关数据
      "avg": float,
      "max": float,
      "min": float
      // 其他心率相关字段
    },
    "respirationRate": {           // 可选，呼吸率相关数据
      "avg": float,
      "max": float,
      "min": float
      // 其他呼吸率相关字段
    },
    "bodyTemp": float,             // 可选，体温
    "movementLevel": float         // 可选，活动水平
  }
  ```
- **响应**: 返回创建的生命体征数据对象
- **状态码**:
  - `200 OK`: 成功
  - `400 Bad Request`: 请求参数错误

### 2.4 更新生命体征数据
- **URL**: `/api/vital-signs/{id}`
- **方法**: `PUT`
- **路径参数**:
  - `id`: 记录ID
- **请求体**: 同添加数据的请求体
- **响应**: 返回更新后的生命体征数据对象
- **状态码**:
  - `200 OK`: 成功
  - `404 Not Found`: 记录不存在
  - `400 Bad Request`: 请求参数错误

### 2.5 删除生命体征数据
- **URL**: `/api/vital-signs/{id}`
- **方法**: `DELETE`
- **路径参数**:
  - `id`: 记录ID
- **响应**: 无响应体
- **状态码**:
  - `200 OK`: 成功删除
  - `404 Not Found`: 记录不存在

---

## 3. 运动轨迹 (Trajectory)

### 3.1 获取所有轨迹数据
- **URL**: `/api/trajectory`
- **方法**: `GET`
- **描述**: 获取所有运动轨迹数据
- **响应格式**: 
  ```json
  [
    {
      "id": long,                   // 记录ID
      "deviceId": "string",         // 设备ID
      "targetId": "string",         // 目标ID
      "timestamp": "ISO时间戳",     
      "position": "json字符串",     // 位置坐标，例如: {"x": 1.2, "y": 3.4}
      "speed": float,               // 移动速度(m/s)
      "direction": float,           // 移动方向(度)
      "createdAt": "ISO时间戳"      // 创建时间
    },
    ...
  ]
  ```

### 3.2 获取指定设备的轨迹数据
- **URL**: `/api/trajectory/device/{deviceId}`
- **方法**: `GET`
- **路径参数**:
  - `deviceId`: 设备唯一标识
- **响应格式**: 同上，返回指定设备的轨迹数据列表，按时间倒序

### 3.3 获取指定设备和目标的轨迹数据
- **URL**: `/api/trajectory/device/{deviceId}/target/{targetId}`
- **方法**: `GET`
- **路径参数**:
  - `deviceId`: 设备唯一标识
  - `targetId`: 目标唯一标识
- **响应格式**: 同上，返回符合条件的轨迹数据列表

### 3.4 添加新的轨迹数据
- **URL**: `/api/trajectory`
- **方法**: `POST`
- **请求体**:
  ```json
  {
    "deviceId": "string",          // 必需，设备ID
    "targetId": "string",          // 可选，目标ID
    "timestamp": "ISO时间戳",      // 可选，默认为当前时间
    "position": {                  // 必需，位置坐标
      "x": float,
      "y": float
      // 可能还有其他坐标信息
    },
    "speed": float,                // 必需，移动速度
    "direction": float             // 必需，移动方向
  }
  ```
- **响应**: 返回创建的轨迹数据对象
- **状态码**:
  - `200 OK`: 成功
  - `400 Bad Request`: 请求参数错误

### 3.5 更新轨迹数据
- **URL**: `/api/trajectory/{id}`
- **方法**: `PUT`
- **路径参数**:
  - `id`: 记录ID
- **请求体**: 同添加数据的请求体
- **响应**: 返回更新后的轨迹数据对象
- **状态码**:
  - `200 OK`: 成功
  - `404 Not Found`: 记录不存在
  - `400 Bad Request`: 请求参数错误

### 3.6 删除轨迹数据
- **URL**: `/api/trajectory/{id}`
- **方法**: `DELETE`
- **路径参数**:
  - `id`: 记录ID
- **响应**: 无响应体
- **状态码**:
  - `200 OK`: 成功删除
  - `404 Not Found`: 记录不存在

---

## 4. 位姿数据 (Pose)

### 4.1 获取所有位姿数据
- **URL**: `/api/pose`
- **方法**: `GET`
- **描述**: 获取所有位姿数据记录
- **响应格式**: 
  ```json
  [
    {
      "id": long,                   // 记录ID
      "deviceId": "string",         // 设备ID
      "targetId": "string",         // 目标ID
      "timestamp": "ISO时间戳",     
      "posture": "string",          // 位姿类型（站立/坐姿/躺姿）
      "angle": float,               // 姿态角度
      "createdAt": "ISO时间戳"      // 创建时间
    },
    ...
  ]
  ```

### 4.2 获取指定设备的位姿数据
- **URL**: `/api/pose/device/{deviceId}`
- **方法**: `GET`
- **路径参数**:
  - `deviceId`: 设备唯一标识
- **响应格式**: 同上，返回指定设备的位姿数据列表

### 4.3 获取指定设备和目标的位姿数据
- **URL**: `/api/pose/device/{deviceId}/target/{targetId}`
- **方法**: `GET`
- **路径参数**:
  - `deviceId`: 设备唯一标识
  - `targetId`: 目标唯一标识
- **响应格式**: 同上，返回符合条件的位姿数据列表

### 4.4 添加新的位姿数据
- **URL**: `/api/pose`
- **方法**: `POST`
- **请求体**:
  ```json
  {
    "deviceId": "string",          // 必需，设备ID
    "targetId": "string",          // 可选，目标ID
    "timestamp": "ISO时间戳",      // 可选，默认为当前时间
    "posture": "string",           // 必需，位姿类型
    "angle": float                 // 可选，姿态角度
  }
  ```
- **响应**: 返回创建的位姿数据对象
- **状态码**:
  - `200 OK`: 成功
  - `400 Bad Request`: 请求参数错误

### 4.5 更新位姿数据
- **URL**: `/api/pose/{id}`
- **方法**: `PUT`
- **路径参数**:
  - `id`: 记录ID
- **请求体**: 同添加数据的请求体
- **响应**: 返回更新后的位姿数据对象
- **状态码**:
  - `200 OK`: 成功
  - `404 Not Found`: 记录不存在
  - `400 Bad Request`: 请求参数错误

### 4.6 删除位姿数据
- **URL**: `/api/pose/{id}`
- **方法**: `DELETE`
- **路径参数**:
  - `id`: 记录ID
- **响应**: 无响应体
- **状态码**:
  - `200 OK`: 成功删除
  - `404 Not Found`: 记录不存在

---

## 5. 心电图数据 (ECG)

### 5.1 获取所有ECG数据
- **URL**: `/api/ecg`
- **方法**: `GET`
- **描述**: 获取所有心电图数据记录
- **响应格式**: 
  ```json
  [
    {
      "id": long,                   // 记录ID
      "deviceId": "string",         // 设备ID
      "targetId": "string",         // 目标ID
      "timestamp": "ISO时间戳",     
      "ecgData": "json字符串",      // ECG数据(JSON数组格式)
      "createdAt": "ISO时间戳"      // 创建时间
    },
    ...
  ]
  ```

### 5.2 获取指定设备的ECG数据
- **URL**: `/api/ecg/device/{deviceId}`
- **方法**: `GET`
- **路径参数**:
  - `deviceId`: 设备唯一标识
- **响应格式**: 同上，返回指定设备的ECG数据列表

### 5.3 获取指定设备和目标的ECG数据
- **URL**: `/api/ecg/device/{deviceId}/target/{targetId}`
- **方法**: `GET`
- **路径参数**:
  - `deviceId`: 设备唯一标识
  - `targetId`: 目标唯一标识
- **响应格式**: 同上，返回符合条件的ECG数据列表

### 5.4 添加新的ECG数据
- **URL**: `/api/ecg`
- **方法**: `POST`
- **请求体**:
  ```json
  {
    "deviceId": "string",          // 必需，设备ID
    "targetId": "string",          // 可选，目标ID
    "timestamp": "ISO时间戳",      // 可选，默认为当前时间
    "ecgData": [                   // 必需，ECG数据点数组
      number,
      number,
      ...
    ]
  }
  ```
- **响应**: 返回创建的ECG数据对象
- **状态码**:
  - `200 OK`: 成功
  - `400 Bad Request`: 请求参数错误

### 5.5 更新ECG数据
- **URL**: `/api/ecg/{id}`
- **方法**: `PUT`
- **路径参数**:
  - `id`: 记录ID
- **请求体**: 同添加数据的请求体
- **响应**: 返回更新后的ECG数据对象
- **状态码**:
  - `200 OK`: 成功
  - `404 Not Found`: 记录不存在
  - `400 Bad Request`: 请求参数错误

### 5.6 删除ECG数据
- **URL**: `/api/ecg/{id}`
- **方法**: `DELETE`
- **路径参数**:
  - `id`: 记录ID
- **响应**: 无响应体
- **状态码**:
  - `200 OK`: 成功删除
  - `404 Not Found`: 记录不存在

---

## 6. WebSocket 实时推送

### 6.1 位姿数据实时推送
- **WebSocket URL**: `/ws/posture`
- **描述**: 通过 SockJS 推送实时位姿/姿态数据
- **连接方式**: 客户端需要使用SockJS库连接，支持跨域订阅
- **推送数据格式**:
  ```json
  {
    "deviceId": "string",         // 设备ID
    "targetId": "string",         // 目标ID
    "timestamp": "ISO时间戳",     
    "posture": "string",          // 位姿类型
    "angle": float                // 姿态角度
  }
  ```
- **应用场景**: 实时监控人体姿态状态，用于健康监测、跌倒预警等

---

*备注：所有接口均支持 CORS，返回 JSON 格式。HTTP状态码按标准定义，特殊逻辑错误会在响应中明确说明。* 