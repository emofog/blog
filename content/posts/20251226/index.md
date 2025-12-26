---
date: '2025-12-26T22:49:38+08:00'
draft: false
title: '在EuRoC数据集上测试ORB_SLAM3效果'
---

> **开发环境**  
> - 操作系统：Windows 11  
> - 虚拟机：VMware Workstation 17.5  
> - 虚拟系统：Ubuntu 20.04.6  

本文记录在 Ubuntu 20.04 虚拟机中搭建 ORB-SLAM3 开发环境并使用 EuRoC 数据集进行测试的完整流程，结合最新官方文档进行了适配和验证。

## 1. 安装 ROS1 及依赖

为简化依赖管理，通过 `fishros` 一键安装 ROS1（Noetic）：

```bash
wget http://fishros.com/install -O fishros && . fishros
```

安装过程中选择 ROS1 版本，并保留默认配置。

安装完成后，补充安装 `ffmpeg`：

```bash
sudo apt install ffmpeg
```

## 2. 安装 Pangolin 0.6

Pangolin 用于可视化，需禁用对 Boost 库的依赖编译安装。

```bash
# 下载源码（建议通过浏览器下载后解压）
# https://github.com/stevenlovegrove/Pangolin/archive/refs/tags/v0.6.zip

cd pangolin-0.6
mkdir build && cd build
cmake -DCPP11_NO_BOOST=1 ..
make
sudo make install
```

## 3. 安装 ORB-SLAM3

### 3.1 安装基础依赖

```bash
sudo apt install git cmake gcc g++ mlocate
```

### 3.2 克隆仓库

```bash
git clone https://github.com/UZ-SLAMLab/ORB_SLAM3.git
```

### 3.3 修改配置以适配环境

1. **检查 OpenCV 版本**：

   ```bash
   pkg-config --modversion opencv4
   ```

2. **修改 `ORB-SLAM3/CMakeLists.txt`**：
   - 将 `find_package(OpenCV X.X)` 改为 `find_package(OpenCV 4.2)`
   - 将 `find_package(Eigen3 X.X REQUIRED)` 改为 `find_package(Eigen3 REQUIRED)`

3. **修改 `build.sh` 中的并行编译参数**：
   - 将所有 `make -j` 行改为合适的参数，注意脚本中有 `make -j4` 避免因核心不足导致编译失败

4. **修改 `Thirdparty/DBoW2/CMakeLists.txt`**：
   - 将 `find_package(OpenCV X.X QUIET)` 改为 `find_package(OpenCV 4.2 QUIET)`

5. **启用可视化窗口**：
   - 编辑 `Examples/Monocular/mono_euroc.cc`
   - 将第 83 行的 `false` 改为 `true`

### 3.4 编译 ORB-SLAM3

```bash
cd ORB-SLAM3
chmod +x build.sh
./build.sh
```

## 4. 下载并组织 EuRoC 数据集

EuRoC 数据集由 ETH 提供[^1]，包含同步的双目图像、IMU 数据及高精度真值[^2]。

1. 访问 [EuRoC MAV Dataset](https://projects.asl.ethz.ch/datasets/euroc-mav/)
2. 下载 `MH_01_easy.zip`（Machine Hall 系列）
3. 在 `ORB-SLAM3` 目录下创建数据集结构 `ORB_SLAM3/datasets/MH01/`
4. 解压 `MH_01_easy.zip` ，将内部的 `mav0` 文件夹移入 `MH01` 文件夹

## 5. 运行测试示例

进入 `Examples` 目录，根据传感器配置运行不同模式。

> 注意：请将路径 `/home/xjc/Desktop/ORB_SLAM3/datasets/MH01` 替换为实际路径。

### 5.1 单目模式

```bash
cd ORB-SLAM3/Examples
./Monocular/mono_euroc ../Vocabulary/ORBvoc.txt ./Monocular/EuRoC.yaml /home/xjc/Desktop/ORB_SLAM3/datasets/MH01 ./Monocular/EuRoC_TimeStamps/MH01.txt
```

### 5.2 单目-惯性模式

```bash
./Monocular-Inertial/mono_inertial_euroc ../Vocabulary/ORBvoc.txt ./Monocular-Inertial/EuRoC.yaml /home/xjc/Desktop/ORB_SLAM3/datasets/MH01 ./Monocular-Inertial/EuRoC_TimeStamps/MH01.txt
```

### 5.3 双目模式

```bash
./Stereo/stereo_euroc ../Vocabulary/ORBvoc.txt ./Stereo/EuRoC.yaml /home/xjc/Desktop/ORB_SLAM3/datasets/MH01 ./Stereo/EuRoC_TimeStamps/MH01.txt
```

### 5.4 双目-惯性模式
```bash
./Stereo-Inertial/stereo_inertial_euroc ../Vocabulary/ORBvoc.txt ./Stereo-Inertial/EuRoC.yaml /home/xjc/Desktop/ORB_SLAM3/datasets/MH01 ./Stereo-Inertial/EuRoC_TimeStamps/MH01.txt dataset-MH01_stereoi
```
注意在 `stereo_inertial_euroc.cc` 源文件中，系统通过以下方式初始化：
```cpp
ORB_SLAM3::System SLAM(argv[1], argv[2], ORB_SLAM3::System::IMU_STEREO, false);
```
其中`false`：表示 **不使用可视化界面（Viewer）**；若设为 `true` 则启用 GUI

[结果视频](https://www.bilibili.com/video/BV1FqBvBbEML/?spm_id_from=333.1387.homepage.video_card.click&vd_source=87511afea40d4a9de7ca2c3b44ff716a)如下：

{{< bilibili BV1FqBvBbEML>}}

---

[^1]: [EuRoC MAV Dataset 官网](https://projects.asl.ethz.ch/datasets/euroc-mav/)  
[^2]: Burri, M., et al. (2016). The EuRoC micro aerial vehicle datasets. *The International Journal of Robotics Research*. [DOI:10.1177/0278364915620033](https://journals.sagepub.com/doi/10.1177/0278364915620033)